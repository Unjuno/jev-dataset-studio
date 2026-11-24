import io
import json
import time
import zipfile

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select, update

from studio.api import create_app
from studio.db import Annotation, Record, Review, WorkItem
from studio.ingest import parse_upload
from studio.providers import DemoProvider, ProviderError
from studio.runtime import CachedProvider, JobInterrupted, RequestBudget
from studio.settings import Settings
from studio.worker import Worker


HEADERS = {"X-Studio-Request": "1"}


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(data_dir=tmp_path))


@pytest.fixture
def client(app):
    with TestClient(app, headers=HEADERS) as client:
        yield client


def sample_and_recipe(client):
    dataset = client.post("/api/datasets/sample").json()
    template = client.get("/api/templates").json()[0]
    return dataset, template["recipe"], template["export_spec"]


def run_job(client, app, recipe=None):
    dataset, default, export = sample_and_recipe(client)
    response = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe or default})
    assert response.status_code == 200, response.text
    job = response.json()
    worker = Worker(app.state.database, app.state.settings)
    while worker.run_once():
        pass
    return client.get(f"/api/jobs/{job['id']}").json(), export


def unzip_response(client, result):
    response = client.get(result["download_url"])
    assert response.status_code == 200
    return zipfile.ZipFile(io.BytesIO(response.content))


def test_preview_batch_review_and_export(client, app):
    dataset, recipe, export = sample_and_recipe(client)
    preview = client.post("/api/preview", json={"dataset_id": dataset["id"], "recipe": recipe, "limit": 2})
    assert preview.status_code == 200, preview.text
    data = preview.json()
    assert data["items"] and data["stats"]["records"] == 2
    assert all(row["origin"] == "demo" and row["status"] == "review" for row in data["items"])
    repeat = client.post("/api/preview", json={"dataset_id": dataset["id"], "recipe": recipe, "limit": 2}).json()
    assert repeat["request_count"] == 0 and repeat["cache_hits"] > 0
    response = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe})
    job_id = response.json()["id"]
    worker = Worker(app.state.database, app.state.settings)
    while worker.run_once():
        pass
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    assert job["completed"] == dataset["rows"] and job["failed"] == 0
    annotations = client.get(f"/api/jobs/{job_id}/annotations").json()["items"]
    for a in annotations:
        assert a["text"][a["target"]["start"]:a["target"]["end"]] == a["target"]["text"]
    a = annotations[0]
    edits = {"phenomenon": "other", "observed": False, "importance": 1.0}
    response = client.patch(f"/api/annotations/{a['id']}", json={
        "revision": a["revision"], "decision": "approved", "features": edits, "note": "確認済み",
    })
    assert response.status_code == 200, response.text
    reviewed = response.json()
    assert reviewed["origin"] == "demo" and reviewed["reviewed"] and reviewed["status"] == "accepted"
    assert reviewed["original_features"] == a["features"] and reviewed["features"] == edits
    assert reviewed["predictions"] == a["predictions"]
    conflict = client.patch(f"/api/annotations/{a['id']}", json={
        "revision": a["revision"], "decision": "rejected",
    })
    assert conflict.status_code == 409
    requests_before = job["request_count"]
    export.update({"selection": "reviewed"})
    response = client.post(f"/api/jobs/{job_id}/exports", json=export)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["rows"] == 1
    archive = unzip_response(client, result)
    row = json.loads(archive.read("dataset.jsonl"))
    assert row["observed"] is False and row["reviewed"] is True
    trace = json.loads(archive.read("provenance.jsonl"))
    assert trace["annotations"][0]["revision"] == 2
    assert trace["annotations"][0]["predictions"] == a["predictions"]
    assert "確認済み" in archive.read("reviews.jsonl").decode()
    assert client.get(f"/api/jobs/{job_id}").json()["request_count"] == requests_before
    # Previously created archives remain immutable after another human revision.
    original_bytes = client.get(result["download_url"]).content
    client.patch(f"/api/annotations/{a['id']}", json={"revision": 2, "decision": "rejected"})
    assert client.get(result["download_url"]).content == original_bytes


def test_modes_and_export_grains(client, app):
    job, spec = run_job(client, app)
    for grain, fmt in [("document", "jsonl"), ("annotation", "csv"), ("target", "parquet")]:
        view = {**spec, "grain": grain, "format": fmt, "columns": [
            {"name": "id", "source": "source_id"}, {"name": "label", "source": "features.phenomenon"},
        ]}
        if grain == "annotation":
            view["columns"] += [{"name": "feature", "source": "feature_id"}, {"name": "value", "source": "value"}]
        response = client.post(f"/api/jobs/{job['id']}/exports", json=view)
        assert response.status_code == 200, response.text
        archive = unzip_response(client, response.json())
        data = archive.read("dataset." + fmt)
        if grain == "document":
            rows = [json.loads(row) for row in data.decode().splitlines()]
            assert len(rows) == 6 and isinstance(rows[0]["label"], list)
        elif fmt == "csv":
            assert data.startswith(b"\xef\xbb\xbf") and b"feature" in data
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pq.read_table(pa.BufferReader(data))
            assert table.num_rows > 6


def test_pause_resume_request_limit_and_cancel(client, app):
    dataset, recipe, _ = sample_and_recipe(client)
    response = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe, "max_requests": 1})
    job_id = response.json()["id"]
    worker = Worker(app.state.database, app.state.settings)
    assert worker.run_once()
    paused = client.get(f"/api/jobs/{job_id}").json()
    assert paused["status"] == "paused" and paused["request_count"] == 1, paused
    assert not worker.run_once()
    response = client.post(f"/api/jobs/{job_id}/resume", json={"max_requests": 1000})
    assert response.status_code == 200
    while worker.run_once():
        pass
    finished = client.get(f"/api/jobs/{job_id}").json()
    assert finished["status"] == "completed" and finished["completed"] == 6
    job = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe}).json()
    assert client.post(f"/api/jobs/{job['id']}/pause").json()["status"] == "paused"
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{job['id']}/resume").status_code == 409


def test_expired_worker_lease_is_fenced_and_reclaimed(client, app):
    dataset, recipe, _ = sample_and_recipe(client)
    job = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe}).json()
    db = app.state.database
    old_worker = Worker(db, app.state.settings)
    item_id, job_id, token = old_worker.claim()
    with db.session() as session:
        session.execute(update(WorkItem).where(WorkItem.id == item_id).values(lease_until=time.time() - 1))
    budget = RequestBudget(db, app.state.settings, job_id=job_id, item_id=item_id, lease=token)
    with pytest.raises(JobInterrupted):
        budget.check()
    restarted = Worker(db, app.state.settings)
    while restarted.run_once():
        pass
    finished = client.get(f"/api/jobs/{job['id']}").json()
    assert finished["status"] == "completed" and finished["completed"] == 6
    with db.session() as session:
        with pytest.raises(JobInterrupted):
            old_worker._item(session, item_id, job_id, token)
        annotations = list(session.scalars(select(Annotation).where(Annotation.job_id == job_id)))
        assert len({(a.record_id, a.target_id) for a in annotations}) == len(annotations)


def test_provider_failure_is_an_error_not_a_label(client, app):
    dataset, recipe, _ = sample_and_recipe(client)
    job = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe}).json()

    class BrokenProvider:
        identity = "broken"

        def evaluate(self, _state, _features):
            raise ProviderError("Provider returned HTTP 401", retryable=False)

    worker = Worker(app.state.database, app.state.settings,
                    provider_factory=lambda *_: (BrokenProvider(), None))
    while worker.run_once():
        pass
    state = client.get(f"/api/jobs/{job['id']}").json()
    assert state["status"] == "completed_with_errors" and state["failed"] == 6
    assert client.get(f"/api/jobs/{job['id']}/annotations").json()["total"] == 0
    assert client.post(f"/api/jobs/{job['id']}/retry-failed").status_code == 200
    worker = Worker(app.state.database, app.state.settings)
    while worker.run_once():
        pass
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "completed"


def test_transient_failures_stop_after_bounded_attempts(client, app):
    dataset, recipe, _ = sample_and_recipe(client)
    job = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe}).json()
    calls = []

    class RateLimited:
        identity = "rate-limited"

        def evaluate(self, _state, _features):
            calls.append(1)
            raise ProviderError("Provider returned HTTP 429", retryable=True)

    worker = Worker(app.state.database, app.state.settings, provider_factory=lambda *_: (RateLimited(), None))
    for _ in range(20):
        if not worker.run_once():
            break
        with app.state.database.session() as session:
            # Advance the durable retry schedule without sleeping through it.
            session.execute(update(WorkItem).where(WorkItem.job_id == job["id"], WorkItem.status == "pending").values(available_at=0))
    final = client.get(f"/api/jobs/{job['id']}").json()
    assert final["status"] == "completed_with_errors" and final["failed"] == 6
    assert len(calls) == 18 and all(e["attempts"] == 3 for e in final["errors"])
    assert client.get(f"/api/jobs/{job['id']}/annotations").json()["items"] == []


def test_cache_does_not_mix_resolved_models_or_cache_aliases(client, app):
    from studio.providers import JevProvider
    from studio.schemas import Recipe
    from studio.targets import build_context, generate_targets
    _, recipe, _ = sample_and_recipe(client)
    recipe = Recipe.model_validate(recipe)
    with app.state.database.session() as session:
        record = session.scalar(select(Record))
    source = record.values["text"]
    state = build_context(source, generate_targets(source, recipe.target)[0], recipe.context)
    provider = JevProvider("test-only-placeholder", "jev-1.14.0")
    resolved = ["jev-1.14.0"]

    def evaluate(context, features):
        answer = DemoProvider(resolved[0]).evaluate(context, features)
        answer.origin = "model"
        return answer

    provider.evaluate = evaluate
    budget = RequestBudget(app.state.database, app.state.settings)
    cached = CachedProvider(provider, app.state.database, "local", budget)
    cached.evaluate(state, recipe.features[:1])
    resolved[0] = "jev-1.15.0"
    with pytest.raises(ProviderError, match="resolved model changed"):
        cached.evaluate(state, recipe.features)
    alias = JevProvider("test-only-placeholder", "jev-latest")
    alias.evaluate = evaluate
    no_cache_budget = RequestBudget(app.state.database, app.state.settings)
    alias_cached = CachedProvider(alias, app.state.database, "local", no_cache_budget)
    alias_cached.evaluate(state, recipe.features)
    alias_cached.evaluate(state, recipe.features)
    assert no_cache_budget.request_count == 2 and no_cache_budget.cache_hits == 0


def test_no_targets_and_invalid_input_are_distinct(client, app):
    dataset, recipe, export = sample_and_recipe(client)
    recipe["target"].update({"mode": "literal", "terms": ["存在しない単語"]})
    job = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe}).json()
    worker = Worker(app.state.database, app.state.settings)
    while worker.run_once():
        pass
    state = client.get(f"/api/jobs/{job['id']}").json()
    assert state["request_count"] == 0 and state["no_target_records"] == 6
    export.update({"grain": "document", "include_empty": True})
    exported = client.post(f"/api/jobs/{job['id']}/exports", json=export).json()
    assert exported["rows"] == 6
    rows = [json.loads(x) for x in unzip_response(client, exported).read("dataset.jsonl").splitlines()]
    assert rows[0]["phenomenon"] == [] and rows[0]["reviewed"] == []
    recipe["source"]["text_field"] = "missing"
    response = client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe})
    assert response.status_code == 400


def test_upload_preserves_unicode_and_nested_data(client):
    source = [{"id": "001", "text": "🔧e\u0301 検査。検査。", "meta": {"tags": ["a", "b"]}}]
    response = client.post("/api/datasets", files={"file": ("source.jsonl", json.dumps(source[0]).encode())})
    assert response.status_code == 200, response.text
    assert response.json()["sample"] == source
    rows = client.get(f"/api/datasets/{response.json()['id']}/records").json()["items"]
    assert rows[0]["values"] == source[0]
    for filename, data in [("bad.csv", b"text,text\na,b"), ("bad.jsonl", b"{\"text\":NaN}"), ("bad.json", b"[]")]:
        assert client.post("/api/datasets", files={"file": (filename, data)}).status_code == 400


def test_live_provider_requires_key(client):
    dataset, recipe, _ = sample_and_recipe(client)
    recipe["provider"] = "jev"
    assert client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe}).status_code == 400
    assert client.get("/api/session").json()["provider_ready"]["jev"] is False


def test_review_rejects_invalid_and_unknown_labels(client, app):
    job, _ = run_job(client, app)
    annotation = client.get(f"/api/jobs/{job['id']}/annotations").json()["items"][0]
    for features in [{"observed": "false"}, {"phenomenon": "invented"}, {"importance": 3}, {"missing": "x"}, {"observed": None}]:
        response = client.patch(f"/api/annotations/{annotation['id']}", json={
            "revision": 1, "decision": "approved", "features": features,
        })
        assert response.status_code == 400, response.text
    with app.state.database.session() as session:
        assert session.get(Annotation, annotation["id"]).revision == 1
        assert not list(session.scalars(select(Review)))


def test_workspace_isolation_and_cross_origin_guard(tmp_path):
    settings = Settings(data_dir=tmp_path, access_tokens={"alice": "a" * 32, "bob": "b" * 32})
    app = create_app(settings)
    with TestClient(app, headers=HEADERS) as alice, TestClient(app, headers=HEADERS) as bob:
        assert alice.get("/api/session").json()["authenticated"] is False
        assert alice.post("/api/datasets/sample").status_code == 401
        assert alice.post("/api/login", json={"token": "wrong"}).status_code == 401
        assert alice.post("/api/login", json={"token": "a" * 32}).status_code == 200
        assert bob.post("/api/login", json={"token": "b" * 32}).status_code == 200
        dataset, recipe, export = sample_and_recipe(alice)
        assert bob.get(f"/api/datasets/{dataset['id']}").status_code == 404
        assert bob.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe}).status_code == 404
        job, export = run_job(alice, app)
        annotations = alice.get(f"/api/jobs/{job['id']}/annotations").json()["items"]
        assert bob.get(f"/api/jobs/{job['id']}/annotations").status_code == 404
        assert bob.patch(f"/api/annotations/{annotations[0]['id']}", json={"revision": 1, "decision": "rejected"}).status_code == 404
        result = alice.post(f"/api/jobs/{job['id']}/exports", json=export).json()
        assert bob.get(result["download_url"]).status_code == 404
        assert bob.post(f"/api/jobs/{job['id']}/exports", json=export).status_code == 404
        assert alice.post("/api/datasets/sample", headers={"Origin": "https://attacker.example"}).status_code == 403
        assert alice.post("/api/datasets/sample", headers={"X-Studio-Request": "0"}).status_code == 403
        assert alice.get("/api/session", headers={"Host": "attacker.example"}).status_code == 400
        assert alice.post("/api/logout").status_code == 200
        assert alice.get("/api/jobs").status_code == 401


def test_cache_is_workspace_scoped_and_export_role_conflicts(client, app):
    _, recipe, _ = sample_and_recipe(client)
    from studio.schemas import Recipe
    typed = Recipe.model_validate(recipe)
    with app.state.database.session() as session:
        record = session.scalar(select(Record))
    from studio.targets import build_context, generate_targets
    state = build_context(record.values["text"], generate_targets(record.values["text"], typed.target)[0], typed.context)
    for owner in ("alice", "alice", "bob"):
        budget = RequestBudget(app.state.database, app.state.settings)
        provider = CachedProvider(DemoProvider(), app.state.database, owner, budget)
        provider.evaluate(state, typed.features)
        if owner == "bob":
            assert budget.request_count == 1
    job, export = run_job(client, app)
    export["columns"] = [{"name": "a", "source": "features.phenomenon", "role": "input"},
                         {"name": "b", "source": "features.phenomenon", "role": "target"}]
    assert client.post(f"/api/jobs/{job['id']}/exports", json=export).status_code == 422


def test_group_and_duplicate_text_splits_and_csv_escape(tmp_path):
    from studio.exports import ExportSpec, SplitSpec, _serialize, record_splits
    from studio.schemas import Recipe
    from studio.templates import builtin_templates
    recipe = Recipe.model_validate(builtin_templates()[0]["recipe"])
    records = [Record(id=str(i), dataset_id="d", position=i, values=row) for i, row in enumerate([
        {"id": "1", "text": "same", "equipment": "A"},
        {"id": "2", "text": "same", "equipment": "B"},
        {"id": "3", "text": "different", "equipment": "B"},
    ])]
    splits = record_splits(records, recipe, SplitSpec(enabled=True))
    assert len(set(splits.values())) == 1
    assert splits == record_splits(list(reversed(records)), recipe, SplitSpec(enabled=True))
    spec = ExportSpec(format="csv", columns=[{"name": "text", "source": "text"}])
    content = _serialize([{"text": "=HYPERLINK(\"example\")"}], spec).decode("utf-8-sig")
    assert "'=HYPERLINK" in content
    # ID-like strings are preserved by the importer.
    settings = Settings(data_dir=tmp_path)
    rows, _, _ = parse_upload("a.csv", b"id,text\n001,hello", settings)
    assert rows[0]["id"] == "001"


def test_export_snapshot_is_stable_while_review_commits(client, app):
    job, _ = run_job(client, app)
    db = app.state.database
    annotation = client.get(f"/api/jobs/{job['id']}/annotations").json()["items"][0]
    # The source snapshot stays at revision 1 while a different connection
    # commits revision 2 and its review history under WAL.
    with db.snapshot() as snapshot:
        old_revision = snapshot.scalar(select(Annotation.revision).where(Annotation.id == annotation["id"]))
        assert old_revision == 1
        result = client.patch(f"/api/annotations/{annotation['id']}", json={"revision": 1, "decision": "rejected"})
        assert result.status_code == 200, result.text
        assert snapshot.scalar(select(Annotation.revision).where(Annotation.id == annotation["id"])) == 1
        assert not list(snapshot.scalars(select(Review).where(Review.annotation_id == annotation["id"])))
    with db.snapshot() as snapshot:
        assert snapshot.scalar(select(Annotation.revision).where(Annotation.id == annotation["id"])) == 2
        assert len(list(snapshot.scalars(select(Review).where(Review.annotation_id == annotation["id"])))) == 1


def create_small_job(client, count=1):
    from studio.templates import builtin_templates
    recipe = builtin_templates()[1]["recipe"]
    rows = [{"text": f"record {i}"} for i in range(count)]
    dataset = client.post("/api/datasets", files={"file": ("rows.json", json.dumps(rows))}).json()
    return client.post("/api/jobs", json={"dataset_id": dataset["id"], "recipe": recipe, "max_requests": 1}).json()


@pytest.mark.parametrize("failed", [False, True])
def test_last_item_and_job_share_crash_boundary(client, app, failed):
    from sqlalchemy import event

    job = create_small_job(client)
    db = app.state.database

    def broken(*_):
        raise ProviderError("Permanent test error.")

    worker = Worker(db, app.state.settings)
    if failed:
        worker.provider_factory = broken
    item_status = "failed" if failed else "completed"

    def flag_terminal_flush(session, *_):
        if any(isinstance(item, WorkItem) and item.status == item_status for item in session.dirty):
            session.info["interrupt_after_commit"] = True

    def simulate_process_exit(session):
        if session.info.pop("interrupt_after_commit", False):
            raise SystemExit("Crash immediately after the terminal work item commits.")

    event.listen(db.sessions, "after_flush", flag_terminal_flush)
    event.listen(db.sessions, "after_commit", simulate_process_exit)
    try:
        with pytest.raises(SystemExit, match="immediately after"):
            worker.run_once()
    finally:
        event.remove(db.sessions, "after_flush", flag_terminal_flush)
        event.remove(db.sessions, "after_commit", simulate_process_exit)
    result = client.get(f"/api/jobs/{job['id']}").json()
    assert result[item_status] == 1
    assert result["status"] == ("completed_with_errors" if failed else "completed")
    assert client.get(f"/api/jobs/{job['id']}/annotations").json()["total"] == (0 if failed else 1)
    assert not worker.run_once()


@pytest.mark.parametrize("action", ["retry-failed", "resume"])
def test_terminal_control_has_no_runnable_orphan_commit(client, app, action):
    from sqlalchemy import event
    from studio.db import Job

    job = create_small_job(client)
    db = app.state.database
    assert Worker(db, app.state.settings).run_once()
    if action == "resume":
        # Previously stopped job whose last work item already committed.
        with db.session() as session:
            session.get(Job, job["id"]).status = "paused"

    committed_states = []

    def observe_commit(_session):
        with db.engine.connect() as connection:
            committed_states.append(connection.scalar(select(Job.status).where(Job.id == job["id"])))

    event.listen(db.sessions, "after_commit", observe_commit)
    try:
        response = client.post(f"/api/jobs/{job['id']}/{action}")
    finally:
        event.remove(db.sessions, "after_commit", observe_commit)
    assert response.status_code == 200 and response.json()["status"] == "completed"
    assert committed_states and set(committed_states) == {"completed"}


@pytest.mark.parametrize("stop", ["pause", "cancel", "pause-resume", "budget"])
def test_stopped_provider_failure_preserves_retry_allowance(client, app, stop):
    count = 2 if stop == "budget" else 1
    job = create_small_job(client, count)
    url = f"/api/jobs/{job['id']}"
    db = app.state.database

    class LateFailure:
        identity = "late-test-failure"

        def evaluate(self, _state, _features):
            if stop == "budget":
                # Our call reserved the only request. The other claim pauses the job.
                assert Worker(db, app.state.settings).run_once()
            else:
                action = "pause" if stop == "pause-resume" else stop
                assert client.post(f"{url}/{action}").status_code == 200
                if stop == "pause-resume":
                    assert client.post(url + "/resume", json={"max_requests": 10}).status_code == 200
            raise ProviderError("Late permanent provider error.")

    def factory(_recipe, _settings, _db, owner, budget):
        return CachedProvider(LateFailure(), db, owner, budget), None

    assert Worker(db, app.state.settings, provider_factory=factory).run_once()
    state = client.get(url).json()
    assert state["pending"] == count and state["failed"] == 0 and not state["errors"]
    with db.session() as session:
        items = list(session.scalars(select(WorkItem).where(WorkItem.job_id == job["id"])))
        assert all(item.attempts == 0 and item.lease_token is None for item in items)
    if stop == "cancel":
        assert state["status"] == "cancelled"
    else:
        if stop != "pause-resume":
            assert state["status"] == "paused"
            assert client.post(url + "/resume", json={"max_requests": 10}).status_code == 200
        worker = Worker(db, app.state.settings)
        while worker.run_once():
            pass
        assert client.get(url).json()["status"] == "completed"


@pytest.mark.parametrize("nested", [False, True])
def test_parquet_upload_rejects_duplicate_field_names_before_conversion(client, nested):
    import pyarrow as pa
    import pyarrow.parquet as pq

    values = [pa.array([1]), pa.array([2])]
    if nested:
        struct = pa.StructArray.from_arrays(values, names=["dup", "dup"])
        lists = pa.ListArray.from_arrays(pa.array([0, 1]), struct)
        table = pa.table({"outer": lists})
    else:
        table = pa.Table.from_arrays(values, names=["dup", "dup"])
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    response = client.post("/api/datasets", files={"file": ("duplicate.parquet", buffer.getvalue())})
    assert response.status_code == 400 and "Duplicate Parquet" in response.json()["detail"]


def test_jsonl_preserves_legal_unicode_separators_with_crlf(client):
    rows = [{"text": "A\u0085B\u2028C\u2029D"}, {"text": "second row"}]
    content = "\r\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\r\n"
    response = client.post("/api/datasets", files={"file": ("unicode.ndjson", content.encode())})
    assert response.status_code == 200 and response.json()["sample"] == rows


def test_review_export_query_uses_bounded_job_and_owner_parameters(client, app):
    from sqlalchemy import event

    job, spec = run_job(client, app)
    url = f"/api/jobs/{job['id']}"
    annotations = client.get(url + "/annotations").json()["items"]
    assert len(annotations) > 3
    reviewed = annotations[0]
    assert client.patch(f"/api/annotations/{reviewed['id']}", json={
        "revision": 1, "decision": "rejected", "note": "included",
    }).status_code == 200
    other_job, _ = run_job(client, app)
    other = client.get(f"/api/jobs/{other_job['id']}/annotations").json()["items"][0]
    assert client.patch(f"/api/annotations/{other['id']}", json={
        "revision": 1, "decision": "rejected", "note": "other job",
    }).status_code == 200
    # A malformed cross-owner historical row must also stay outside this workspace.
    with app.state.database.session() as session:
        session.add(Review(annotation_id=reviewed["id"], owner="someone-else", previous_revision=9,
                           decision="rejected", features={}, note="other owner"))
    bind_counts = []

    def capture(_conn, _cursor, statement, parameters, *_):
        if "from reviews" in statement.lower():
            bind_counts.append(len(parameters))

    engine = app.state.database.engine
    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = client.post(url + "/exports", json=spec).json()
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert bind_counts and max(bind_counts) <= 3
    reviews = unzip_response(client, result).read("reviews.jsonl").splitlines()
    assert [json.loads(line)["note"] for line in reviews] == ["included"]


@pytest.mark.parametrize("grain", ["target", "document"])
def test_parquet_folds_share_known_and_inferred_types(client, app, grain):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from studio.exports import SplitSpec, record_splits
    from studio.schemas import Recipe

    job, _ = run_job(client, app)
    split = SplitSpec(enabled=True, train=0.5, validation=0.5, test=0)
    with app.state.database.session() as session:
        records = list(session.scalars(select(Record).where(Record.dataset_id == job["dataset_id"])))
        assignment = record_splits(records, Recipe.model_validate(job["recipe"]), split)
        assert set(assignment.values()) == {"train", "validation"}
        for a in session.scalars(select(Annotation).where(Annotation.job_id == job["id"])):
            value = False if assignment[a.record_id] == "train" else None
            a.features = {**a.features, "observed": value, "importance": None}
    spec = {"format": "parquet", "grain": grain, "split": split.model_dump(), "columns": [
        {"name": "flag", "source": "features.observed"}, {"name": "score", "source": "features.importance"},
        {"name": "equipment", "source": "source.equipment"}, {"name": "start", "source": "target.start"},
    ]}
    response = client.post(f"/api/jobs/{job['id']}/exports", json=spec)
    assert response.status_code == 200, response.text
    archive = unzip_response(client, response.json())
    train, validation, empty = [
        pq.read_table(pa.BufferReader(archive.read(name + ".parquet"))) for name in ("train", "validation", "test")
    ]
    shape = pa.list_ if grain == "document" else lambda t: t
    expected = pa.schema([("flag", shape(pa.bool_())), ("score", shape(pa.float64())),
                          ("equipment", pa.string()), ("start", shape(pa.int64()))])
    assert train.schema == validation.schema == empty.schema == expected
    assert train.num_rows and validation.num_rows and not empty.num_rows
    for table, value in [(train, False), (validation, None)]:
        assert all((set(row) == {value} if grain == "document" else row is value)
                   for row in table["flag"].to_pylist())
    assert pa.concat_tables([train, validation, empty]).num_rows == response.json()["rows"]


@pytest.mark.parametrize("shortcut,field", [("text", "text"), ("source_id", "id")])
def test_export_rejects_recipe_resolved_role_aliases(client, app, shortcut, field):
    job, spec = run_job(client, app)
    spec["columns"] = [{"name": "left", "source": "source." + field, "role": "input"},
                       {"name": "right", "source": shortcut, "role": "target"}]
    response = client.post(f"/api/jobs/{job['id']}/exports", json=spec)
    assert response.status_code == 400 and "both an input and a target" in response.json()["detail"]
    spec["columns"][1]["role"] = "metadata"
    assert client.post(f"/api/jobs/{job['id']}/exports", json=spec).status_code == 200


@pytest.mark.parametrize("left,right", [("text", "target.text"), ("value", "features.label")])
def test_annotation_view_resolves_document_and_single_feature_aliases(client, left, right):
    job = create_small_job(client)
    spec = {"grain": "annotation", "columns": [{"name": "a", "source": left, "role": "target"},
                                                {"name": "b", "source": right, "role": "input"}]}
    response = client.post(f"/api/jobs/{job['id']}/exports", json=spec)
    assert response.status_code == 400 and "both an input and a target" in response.json()["detail"]


def test_export_column_order_and_csv_header_guard(client, app):
    import csv

    job, spec = run_job(client, app)
    url = f"/api/jobs/{job['id']}/exports"
    spec["columns"] = [{"name": "z", "source": "text"}, {"name": "  =label", "source": "reviewed"},
                       {"name": "a", "source": "source_id"}]
    result = client.post(url, json=spec).json()
    row = json.loads(unzip_response(client, result).read("dataset.jsonl").splitlines()[0])
    assert list(row) == ["z", "  =label", "a"]
    for escape, expected in [(True, "'  =label"), (False, "  =label")]:
        result = client.post(url, json={**spec, "format": "csv", "csv_escape_formulas": escape}).json()
        data = unzip_response(client, result).read("dataset.csv").decode("utf-8-sig")
        assert next(csv.reader(io.StringIO(data))) == ["z", expected, "a"]
    spec["columns"][0]["name"] = "'  =label"
    assert client.post(url, json={**spec, "format": "csv"}).status_code == 400
