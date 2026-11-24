"""Materialize immutable dataset views without invoking an inference provider."""

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from typing import Literal
import zipfile

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from .db import Annotation, Dataset, Export, Job, Record, Review, WorkItem, uid
from .ingest import source_id
from .schemas import Recipe, canonical_hash


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


class Column(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=250)
    role: Literal["input", "target", "metadata", "group"] = "metadata"


class SplitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    train: float = Field(default=0.8, ge=0, le=1)
    validation: float = Field(default=0.1, ge=0, le=1)
    test: float = Field(default=0.1, ge=0, le=1)
    seed: str = Field(default="42", max_length=100)

    @model_validator(mode="after")
    def ratios(self):
        if not math.isclose(self.train + self.validation + self.test, 1, abs_tol=1e-9):
            raise ValueError("Split ratios must sum to 1.")
        return self


class ExportSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["jsonl", "csv", "parquet"] = "jsonl"
    grain: Literal["target", "document", "annotation"] = "target"
    selection: Literal["accepted", "reviewed", "all"] = "all"
    columns: list[Column] = Field(min_length=1, max_length=64)
    split: SplitSpec = Field(default_factory=SplitSpec)
    include_empty: bool = False
    csv_escape_formulas: bool = True

    @model_validator(mode="after")
    def valid_columns(self):
        if len({c.name for c in self.columns}) != len(self.columns):
            raise ValueError("Output column names must be unique.")
        inputs = {c.source for c in self.columns if c.role == "input"}
        targets = {c.source for c in self.columns if c.role == "target"}
        if inputs & targets:
            raise ValueError("The same source value cannot be both an input and a target column.")
        if self.include_empty and self.grain != "document":
            raise ValueError("Empty-source rows can only be included in document grain.")
        return self


def record_splits(records: list, recipe: Recipe, spec: SplitSpec) -> dict[str, str]:
    """Connected components unite explicit groups AND exact duplicate texts."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            next_key = parent[x]
            parent[x] = root
            x = next_key
        return root

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    keys = {}
    for record in records:
        text = record.values[recipe.source.text_field]
        text_key = "text:" + hashlib.sha256(text.encode()).hexdigest()
        group_key = ("group:" + str(record.values[recipe.source.group_field])) if recipe.source.group_field else text_key
        union(group_key, text_key)
        keys[record.id] = group_key
    result = {}
    for record in records:
        group = find(keys[record.id])
        number = int(hashlib.sha256((spec.seed + "\0" + group).encode()).hexdigest(), 16) / 2**256
        result[record.id] = ("train" if number < spec.train else
                             "validation" if number < spec.train + spec.validation else "test")
    return result


def _matches(annotation, selection):
    if selection == "all":
        return True
    if selection == "reviewed":
        return annotation.reviewed and annotation.status == "accepted"
    return annotation.status == "accepted"


def _column_value(column, record, annotations, recipe, split, grain, feature_id=None):
    source = column.source
    if source == "source_id":
        return source_id(record, recipe)
    if source == "text":
        return record.values[recipe.source.text_field]
    if source == "split":
        return split
    if source.startswith("source."):
        return record.values.get(source[7:])
    if source == "feature_id":
        return feature_id
    if source == "value":
        return annotations[0].features[feature_id]
    if source.startswith("target."):
        values = [a.target[source[7:]] for a in annotations]
    elif source.startswith("features."):
        values = [a.features.get(source[9:]) for a in annotations]
    elif source in {"status", "origin", "reviewed"}:
        values = [getattr(a, source) for a in annotations]
    else:
        raise ValueError(f"Unknown output source: {source}")
    # Document-level annotation columns always have a stable array type.
    return values if grain == "document" else values[0]


def _serialize(rows: list[dict], spec: ExportSpec) -> bytes:
    if spec.format == "jsonl":
        return "".join(json_text(r) + "\n" for r in rows).encode()
    if spec.format == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        sink = pa.BufferOutputStream()
        try:
            table = pa.Table.from_pylist(rows) if rows else pa.table({c.name: [] for c in spec.columns})
            pq.write_table(table, sink)
        except (pa.ArrowException, TypeError, ValueError) as error:
            raise ValueError("This view contains incompatible column types for Parquet. Use JSONL or choose consistent fields.") from error
        return sink.getvalue().to_pybytes()
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=[c.name for c in spec.columns], lineterminator="\n")
    writer.writeheader()
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                value = json_text(value)
            if spec.csv_escape_formulas and isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return output.getvalue().encode("utf-8-sig")


def create_export(database, settings, owner: str, job_id: str, spec: ExportSpec) -> dict:
    # A repeatable snapshot keeps annotations, review history, and job state aligned
    # even when workers or reviewers finish while an archive is being assembled.
    with database.snapshot() as session:
        job = session.scalar(select(Job).where(Job.id == job_id, Job.owner == owner))
        if not job:
            raise LookupError("Job not found.")
        recipe = Recipe.model_validate(job.recipe)
        dataset = session.get(Dataset, job.dataset_id)
        records = list(session.scalars(select(Record).where(Record.dataset_id == job.dataset_id).order_by(Record.position)))
        query = select(Annotation).where(Annotation.job_id == job_id).order_by(Annotation.record_id, Annotation.target["start"].as_integer())
        annotations = list(session.scalars(query))
        processing = {w.record_id: w for w in session.scalars(select(WorkItem).where(WorkItem.job_id == job_id))}
        reviews = list(session.scalars(select(Review).where(Review.annotation_id.in_([a.id for a in annotations])))) if annotations else []
        # All scalar/JSON objects below are detached snapshot data, not lazy DB relationships.
        session.expunge_all()
    allowed = {"source_id", "text", "target.text", "target.start", "target.end", "status", "origin", "reviewed", "split"}
    allowed |= {"features." + f.id for f in recipe.features}
    allowed |= {"source." + col for col in dataset.columns}
    if spec.grain == "annotation":
        allowed |= {"feature_id", "value"}
    for column in spec.columns:
        if column.source not in allowed:
            raise ValueError(f"Output source is not available in this view: {column.source}")
    splits = record_splits(records, recipe, spec.split) if spec.split.enabled else {r.id: "dataset" for r in records}
    grouped = defaultdict(list)
    for annotation in annotations:
        if _matches(annotation, spec.selection):
            grouped[annotation.record_id].append(annotation)
    data = defaultdict(list)
    provenance = []
    for record in records:
        selected = grouped[record.id]
        if spec.grain == "document":
            units = [(selected, None)] if selected or spec.include_empty else []
        elif spec.grain == "target":
            units = [([a], None) for a in selected]
        else:
            units = [([a], f.id) for a in selected for f in recipe.features]
        for unit, feature_id in units:
            split = splits[record.id]
            row = {c.name: _column_value(c, record, unit, recipe, split, spec.grain, feature_id) for c in spec.columns}
            data[split].append(row)
            provenance.append({
                "file": f"{split}.{spec.format}", "row": len(data[split]),
                "output_sha256": hashlib.sha256(json_text(row).encode()).hexdigest(),
                "source_id": source_id(record, recipe), "record_id": record.id,
                "dataset_id": dataset.id, "source_sha256": dataset.checksum,
                "processing_status": processing[record.id].status,
                "processing_error": processing[record.id].error,
                "no_targets": bool(processing[record.id].result.get("no_targets")),
                "annotations": [{"id": a.id, "revision": a.revision, "target": a.target,
                                 "origin": a.origin, "reviewed": a.reviewed, "status": a.status,
                                 "features": a.features, "predictions": a.predictions,
                                 "reasons": a.reasons} for a in unit],
                "feature_id": feature_id,
            })
    if spec.split.enabled:
        for split in ("train", "validation", "test"):
            data.setdefault(split, [])
    else:
        data.setdefault("dataset", [])
    export_id = uid()
    files = {f"{split}.{spec.format}": _serialize(rows, spec) for split, rows in data.items()}
    manifest = {
        "version": 1, "export_id": export_id, "job_id": job_id, "dataset_id": dataset.id,
        "dataset_name": dataset.name, "source_sha256": dataset.checksum,
        "recipe_sha256": canonical_hash(recipe), "export_spec": spec.model_dump(mode="json"),
        "job_status_at_snapshot": job.status, "row_counts": {s: len(r) for s, r in data.items()},
        "annotation_counts": dict(Counter(a.status for a in annotations)),
        "processing_counts": dict(Counter(w.status for w in processing.values())),
        "demo_annotations": sum(a.origin == "demo" for a in annotations),
        "human_reviewed": sum(a.reviewed for a in annotations),
        "annotation_revisions": {a.id: a.revision for a in annotations},
        "models": sorted({str(p.get("model", "unknown")) for a in annotations for p in a.predictions}),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        "quality_note": "Automatic acceptance is not a human correctness certificate. Demo labels are illustrative. Empty documents are not certified negatives.",
        "split_note": "Explicit source groups and identical source texts stay together. Ratios are expected hash proportions, not exact counts. Near-duplicate detection is not guaranteed.",
    }
    files["provenance.jsonl"] = "".join(json_text(p) + "\n" for p in provenance).encode()
    files["reviews.jsonl"] = "".join(json_text({
        "annotation_id": r.annotation_id, "previous_revision": r.previous_revision,
        "decision": r.decision, "features": r.features, "note": r.note, "created_at": r.created_at,
    }) + "\n" for r in reviews).encode()
    files["recipe.json"] = json.dumps(recipe.model_dump(mode="json"), ensure_ascii=False, indent=2).encode()
    manifest["files"] = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    directory = settings.data_dir / "exports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (export_id + ".zip")
    temporary = directory / (export_id + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, content in files.items():
                archive.writestr(filename, content)
        os.replace(temporary, path)
        with database.session() as session:
            session.add(Export(id=export_id, owner=owner, job_id=job_id, path=str(path), manifest=manifest))
    except Exception:
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        raise
    return {"id": export_id, "download_url": f"/api/exports/{export_id}/download",
            "rows": sum(len(rows) for rows in data.values()), "manifest": manifest}
