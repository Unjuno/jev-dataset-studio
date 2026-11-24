"""HTTP boundary: ownership, typed configuration, immutable inputs, and reviews."""

from contextlib import asynccontextmanager
import csv
import hmac
import json
import logging
from pathlib import Path
import time
from typing import Literal

from fastapi import APIRouter, Body, Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select, text, update
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .auth import COOKIE, require_same_origin, sign_session, workspace
from .db import Annotation, Database, Dataset, Export, Job, Record, Review, Template, WorkItem, uid
from .exports import ExportSpec, create_export
from .ingest import parse_upload, validate_source
from .providers import ProviderError
from .runtime import BudgetExceeded, RequestBudget, ensure_providers, make_providers, process_record
from .schemas import Recipe
from .settings import Settings
from .templates import SAMPLE, builtin_templates
from .worker import finish_job, stop_job

logger = logging.getLogger(__name__)


class RequestSizeLimit:
    """Bound streamed/chunked bodies before multipart spooling as well."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.limit:
                    raise HTTPException(413, "Request body exceeds the server size limit.")
            return message

        return await self.app(scope, bounded_receive, send)


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Login(InputModel):
    token: str = Field(min_length=1, max_length=500)


class TemplateInput(InputModel):
    name: str = Field(min_length=1, max_length=200)
    recipe: Recipe
    export_spec: ExportSpec


class PreviewInput(InputModel):
    dataset_id: str
    recipe: Recipe
    limit: int = Field(default=3, ge=1, le=20)
    max_requests: int = Field(default=100, ge=1, le=1000)


class JobInput(InputModel):
    dataset_id: str
    recipe: Recipe
    max_requests: int = Field(default=10000, ge=1, le=1000000)
    budget_usd: float | None = Field(default=None, gt=0)


class ResumeInput(InputModel):
    max_requests: int | None = Field(default=None, ge=1, le=1000000)
    budget_usd: float | None = Field(default=None, gt=0)


class ReviewInput(InputModel):
    revision: int = Field(ge=1)
    decision: Literal["approved", "rejected"]
    features: dict = Field(default_factory=dict)
    note: str = Field(default="", max_length=4000)


def dataset_summary(dataset):
    return {key: getattr(dataset, key) for key in ("id", "name", "rows", "columns", "created_at")}


def owned(session, model, identifier, owner, *, lock=False):
    query = select(model).where(model.id == identifier, model.owner == owner)
    item = session.scalar(query.with_for_update() if lock else query)
    if not item:
        raise HTTPException(404, "対象が見つかりません / Not found in this workspace.")
    return item


def job_summary(session, job):
    counts = dict(session.execute(select(WorkItem.status, func.count()).where(
        WorkItem.job_id == job.id,
    ).group_by(WorkItem.status)).all())
    errors = [{"record_id": item.record_id, "message": item.error, "attempts": item.attempts,
               "status": item.status} for item in session.scalars(select(WorkItem).where(
                   WorkItem.job_id == job.id, WorkItem.error != "",
               ).limit(50))]
    no_targets = sum(bool(item.result.get("no_targets")) for item in session.scalars(select(WorkItem).where(
        WorkItem.job_id == job.id, WorkItem.status == "completed",
    )))
    result = {key: getattr(job, key) for key in (
        "id", "dataset_id", "recipe", "status", "total", "request_count", "max_requests", "budget_usd",
        "estimated_cost_usd", "actual_input_tokens", "message", "created_at", "updated_at",
    )}
    result.update({"completed": counts.get("completed", 0), "failed": counts.get("failed", 0),
                   "running": counts.get("running", 0), "pending": counts.get("pending", 0),
                   "no_target_records": no_targets, "errors": errors})
    return result


def annotation_summary(session, annotation, recipe=None):
    if recipe is None:
        recipe = Recipe.model_validate(session.get(Job, annotation.job_id).recipe)
    record = session.get(Record, annotation.record_id)
    result = {key: getattr(annotation, key) for key in (
        "id", "job_id", "record_id", "source_id", "target", "features", "original_features", "predictions",
        "reasons", "status", "origin", "reviewed", "revision", "created_at",
    )}
    result["text"] = record.values[recipe.source.text_field]
    return result


def create_app(settings=None):
    settings = settings or Settings.from_env()
    database = Database(settings)

    @asynccontextmanager
    async def lifespan(_app):
        database.initialize()
        yield
        database.engine.dispose()

    app = FastAPI(title="Jev Dataset Studio", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings, app.state.database = settings, database
    app.add_middleware(RequestSizeLimit, limit=settings.max_upload_bytes + 1024 * 1024)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

    @app.middleware("http")
    async def response_policy(request, call_next):
        # Reject oversized requests before multipart parsing (also bounded while reading).
        length = request.headers.get("content-length")
        try:
            if length and int(length) > settings.max_upload_bytes + 1024 * 1024:
                return JSONResponse({"detail": "Upload exceeds the server size limit."}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, error):
        return JSONResponse({"detail": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                                       for e in error.errors()]}, status_code=422)

    @app.exception_handler(ValueError)
    async def value_error(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(LookupError)
    async def lookup_error(_request, _error):
        return JSONResponse({"detail": "Not found."}, status_code=404)

    api = APIRouter(prefix="/api", dependencies=[Depends(require_same_origin)])

    @api.get("/health")
    def health():
        with database.session() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok", "database": "ok"}

    @api.get("/session")
    def get_session(request: Request):
        try:
            owner = workspace(request)
        except HTTPException:
            owner = None
        return {"authenticated": owner is not None, "auth_required": bool(settings.access_tokens),
                "workspace": owner, "provider_ready": {"jev": bool(settings.jev_api_key), "verifier": settings.verifier_ready},
                "default_model": settings.jev_model,
                "limits": {"upload_bytes": settings.max_upload_bytes, "rows": settings.max_rows,
                           "text_chars": settings.max_text_chars}}

    @api.post("/login")
    def login(body: Login, response: Response):
        for name, token in settings.access_tokens.items():
            if hmac.compare_digest(body.token.encode(), token.encode()):
                response.set_cookie(COOKIE, sign_session(settings, name), httponly=True, samesite="strict",
                                    secure=settings.secure_cookies, max_age=12 * 3600, path="/")
                return {"authenticated": True, "workspace": name}
        raise HTTPException(401, "トークンが一致しません / Invalid workspace token.")

    @api.post("/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE, path="/")
        return {"authenticated": False}

    def save_dataset(owner, filename, content):
        try:
            rows, columns, checksum = parse_upload(filename, content, settings)
        except csv.Error as error:
            raise ValueError("Invalid CSV or a CSV field exceeds the supported size.") from error
        dataset_id = uid()
        directory = settings.data_dir / "sources"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / dataset_id
        path.write_bytes(content)
        try:
            with database.session() as session:
                dataset = Dataset(id=dataset_id, owner=owner, name=Path(filename).name[:250], rows=len(rows),
                                  columns=columns, checksum=checksum, original_path=str(path))
                session.add(dataset)
                session.add_all([Record(dataset_id=dataset_id, position=i, values=row) for i, row in enumerate(rows)])
                session.flush()
                return {**dataset_summary(dataset), "sample": rows[:10]}
        except Exception:
            path.unlink(missing_ok=True)
            raise

    @api.get("/datasets")
    def list_datasets(owner=Depends(workspace)):
        with database.session() as session:
            return [dataset_summary(d) for d in session.scalars(select(Dataset).where(
                Dataset.owner == owner,
            ).order_by(Dataset.created_at.desc()))]

    @api.post("/datasets")
    async def upload_dataset(file: UploadFile = File(...), owner=Depends(workspace)):
        try:
            content = await file.read(settings.max_upload_bytes + 1)
            if len(content) > settings.max_upload_bytes:
                raise HTTPException(413, "アップロード上限を超えています / Upload too large.")
            return save_dataset(owner, file.filename or "dataset.jsonl", content)
        finally:
            await file.close()

    @api.post("/datasets/sample")
    def sample_dataset(owner=Depends(workspace)):
        return save_dataset(owner, "sample-equipment.jsonl", "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in SAMPLE
        ).encode())

    @api.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, owner=Depends(workspace)):
        with database.session() as session:
            dataset = owned(session, Dataset, dataset_id, owner)
            sample = [r.values for r in session.scalars(select(Record).where(
                Record.dataset_id == dataset_id,
            ).order_by(Record.position).limit(10))]
            return {**dataset_summary(dataset), "sample": sample}

    @api.get("/datasets/{dataset_id}/records")
    def list_records(dataset_id: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100), owner=Depends(workspace)):
        with database.session() as session:
            dataset = owned(session, Dataset, dataset_id, owner)
            rows = session.scalars(select(Record).where(Record.dataset_id == dataset_id).order_by(
                Record.position,
            ).offset(offset).limit(limit))
            return {"items": [{"id": r.id, "position": r.position, "values": r.values} for r in rows], "total": dataset.rows}

    @api.get("/templates")
    def list_templates(owner=Depends(workspace)):
        with database.session() as session:
            rows = [{"id": t.id, "name": t.name, "recipe": t.recipe, "export_spec": t.export_spec,
                     "created_at": t.created_at, "builtin": False} for t in session.scalars(select(Template).where(
                         Template.owner == owner,
                     ).order_by(Template.created_at.desc()))]
            return builtin_templates(settings.jev_model) + rows

    @api.post("/templates")
    def save_template(body: TemplateInput, owner=Depends(workspace)):
        body.export_spec.validate_roles(body.recipe)
        with database.session() as session:
            template = Template(owner=owner, name=body.name, recipe=body.recipe.model_dump(mode="json"),
                                export_spec=body.export_spec.model_dump(mode="json"))
            session.add(template)
            session.flush()
            return {"id": template.id, "name": template.name, "recipe": template.recipe,
                    "export_spec": template.export_spec, "created_at": template.created_at, "builtin": False}

    def input_records(dataset_id, owner, recipe):
        with database.session() as session:
            owned(session, Dataset, dataset_id, owner)
            records = list(session.scalars(select(Record).where(Record.dataset_id == dataset_id).order_by(Record.position)))
        validate_source(records, recipe, settings)
        return records

    @api.post("/preview")
    def preview(body: PreviewInput, owner=Depends(workspace)):
        records = input_records(body.dataset_id, owner, body.recipe)
        budget = RequestBudget(database, settings, max_requests=body.max_requests)
        primary, verifier = make_providers(body.recipe, settings, database, owner, budget)
        items, errors = [], []
        completed, no_targets = 0, 0
        for record in records[:body.limit]:
            try:
                result = process_record(record, body.recipe, primary, verifier)
                for annotation in result:
                    annotation.update({"id": "preview-" + uid(), "revision": 0,
                                       "original_features": annotation["features"]})
                items.extend(result)
                no_targets += not result
                completed += 1
            except (ProviderError, ValueError, BudgetExceeded) as error:
                errors.append({"record_id": record.id, "message": str(error)})
                if isinstance(error, BudgetExceeded):
                    break
        counts = {status: sum(a["status"] == status for a in items) for status in ("accepted", "review", "excluded")}
        return {"items": items, "errors": errors,
                "stats": {**counts, "records": completed, "annotations": len(items), "no_target_records": no_targets},
                "request_count": budget.request_count, "cache_hits": budget.cache_hits,
                "estimated_cost_usd": budget.estimated_cost_usd, "actual_input_tokens": budget.actual_input_tokens}

    @api.post("/jobs")
    def create_job(body: JobInput, owner=Depends(workspace)):
        ensure_providers(body.recipe, settings)
        records = input_records(body.dataset_id, owner, body.recipe)
        with database.session() as session:
            job = Job(owner=owner, dataset_id=body.dataset_id, recipe=body.recipe.model_dump(mode="json"),
                      total=len(records), max_requests=body.max_requests, budget_usd=body.budget_usd)
            session.add(job)
            session.flush()
            session.add_all([WorkItem(job_id=job.id, record_id=r.id) for r in records])
            session.flush()
            return job_summary(session, job)

    @api.get("/jobs")
    def list_jobs(owner=Depends(workspace)):
        with database.session() as session:
            return [job_summary(session, j) for j in session.scalars(select(Job).where(Job.owner == owner).order_by(
                Job.created_at.desc(),
            ).limit(100))]

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str, owner=Depends(workspace)):
        with database.session() as session:
            return job_summary(session, owned(session, Job, job_id, owner))

    @api.post("/jobs/{job_id}/{action}")
    def control_job(job_id: str, action: str, body: ResumeInput | None = Body(default=None), owner=Depends(workspace)):
        if action not in {"pause", "resume", "cancel", "retry-failed"}:
            raise HTTPException(404, "Unknown job action.")
        with database.session() as session:
            job = owned(session, Job, job_id, owner, lock=True)
            if action == "pause" and job.status in {"queued", "running"}:
                stop_job(session, job, "paused")
            elif action == "cancel" and job.status in {"queued", "running", "paused"}:
                stop_job(session, job, "cancelled")
            elif action == "resume" and job.status == "paused":
                if body and body.max_requests is not None:
                    job.max_requests = body.max_requests
                if body and body.budget_usd is not None:
                    job.budget_usd = body.budget_usd
                job.status, job.message = "queued", ""
            elif action == "retry-failed" and job.status in {"completed_with_errors", "paused", "completed"}:
                session.execute(update(WorkItem).where(WorkItem.job_id == job_id, WorkItem.status == "failed").values(
                    status="pending", attempts=0, available_at=0, error="", lease_token=None, lease_until=0,
                ))
                job.status, job.message = "queued", ""
            else:
                raise HTTPException(409, "この状態では実行できません / Action is not valid in the current job state.")
            job.updated_at = time.time()
            finish_job(session, job)
            return job_summary(session, job)

    @api.get("/jobs/{job_id}/annotations")
    def list_annotations(job_id: str, status: Literal["all", "accepted", "review", "excluded"] = "all",
                         offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), owner=Depends(workspace)):
        with database.session() as session:
            job = owned(session, Job, job_id, owner)
            recipe = Recipe.model_validate(job.recipe)
            conditions = [Annotation.job_id == job_id]
            if status != "all":
                conditions.append(Annotation.status == status)
            total = session.scalar(select(func.count()).select_from(Annotation).where(*conditions))
            rows = list(session.scalars(select(Annotation).join(Record, Record.id == Annotation.record_id).where(
                *conditions,
            ).order_by(Record.position, Annotation.target["start"].as_integer()).offset(offset).limit(limit)))
            counts = dict(session.execute(select(Annotation.status, func.count()).where(
                Annotation.job_id == job_id,
            ).group_by(Annotation.status)).all())
            return {"items": [annotation_summary(session, a, recipe) for a in rows], "total": total, "counts": counts}

    @api.patch("/annotations/{annotation_id}")
    def review_annotation(annotation_id: str, body: ReviewInput, owner=Depends(workspace)):
        with database.session() as session:
            annotation = owned(session, Annotation, annotation_id, owner, lock=True)
            recipe = Recipe.model_validate(session.get(Job, annotation.job_id).recipe)
            allowed = {f.id: f for f in recipe.features}
            if set(body.features) - set(allowed):
                raise ValueError("Review contains an unknown feature ID.")
            features = {**annotation.features, **body.features}
            for key, value in features.items():
                allowed[key].validate_value(value)
            if body.decision == "approved" and any(value is None for value in features.values()):
                raise ValueError("未確定の特徴があります。値を確定するか除外してください / Resolve unknown features before approval.")
            result = session.execute(update(Annotation).where(
                Annotation.id == annotation_id, Annotation.owner == owner, Annotation.revision == body.revision,
            ).values(features=features, reviewed=True, status="accepted" if body.decision == "approved" else "excluded",
                     revision=Annotation.revision + 1), execution_options={"synchronize_session": False})
            if result.rowcount != 1:
                raise HTTPException(409, "別の編集が反映されています。再読込してください / Review revision changed.")
            session.add(Review(annotation_id=annotation_id, owner=owner, previous_revision=body.revision,
                               decision=body.decision, features=features, note=body.note))
            session.expire(annotation)
            return annotation_summary(session, annotation, recipe)

    # Specific endpoints are ordered before the generic action route below.
    @api.post("/jobs/{job_id}/exports")
    def export_job(job_id: str, body: ExportSpec, owner=Depends(workspace)):
        return create_export(database, settings, owner, job_id, body)

    @api.get("/exports/{export_id}/download")
    def download(export_id: str, owner=Depends(workspace)):
        with database.session() as session:
            item = owned(session, Export, export_id, owner)
            path = Path(item.path)
        if not path.is_file():
            raise HTTPException(410, "Export file is unavailable. Create the export again.")
        return FileResponse(path, media_type="application/zip", filename=f"jev-dataset-{export_id[:8]}.zip")

    # FastAPI matches in registration order; literal endpoints precede the generic action endpoint.
    api.routes.sort(key=lambda route: "{action}" in route.path)
    app.include_router(api)
    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="static")
    return app
