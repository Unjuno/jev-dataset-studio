"""Opt-in PostgreSQL integration; CI supplies an isolated test database."""

from concurrent.futures import ThreadPoolExecutor
import os
import time

import pytest
from sqlalchemy import create_engine, select, text, update
from sqlalchemy.engine import make_url

from studio.db import Annotation, Database, Dataset, Job, Record, WorkItem, uid
from studio.settings import Settings
from studio.templates import SAMPLE, builtin_templates
from studio.worker import Worker


@pytest.fixture
def postgres(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL for PostgreSQL integration (enabled in CI).")
    schema = "test_studio_" + uid()
    admin = create_engine(url)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    scoped = make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    settings = Settings(data_dir=tmp_path, database_url=scoped.render_as_string(hide_password=False))
    database = Database(settings)
    database.initialize()
    try:
        yield database, settings
    finally:
        database.engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def test_parallel_workers_have_distinct_leases_and_single_results(postgres):
    database, settings = postgres
    with database.session() as session:
        dataset = Dataset(owner="local", name="parallel", columns=["id", "text", "equipment"],
                          rows=6, checksum="test", original_path="synthetic")
        session.add(dataset)
        session.flush()
        records = [Record(dataset_id=dataset.id, position=i, values=row) for i, row in enumerate(SAMPLE)]
        session.add_all(records)
        session.flush()
        job = Job(owner="local", dataset_id=dataset.id, recipe=builtin_templates()[0]["recipe"], total=6)
        session.add(job)
        session.flush()
        job_id = job.id
        session.add_all([WorkItem(job_id=job.id, record_id=r.id) for r in records])
    workers = [Worker(database, settings) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        claims = list(pool.map(lambda worker: worker.claim(), workers))
    assert len({claim[0] for claim in claims}) == 3
    with database.session() as session:
        session.execute(update(WorkItem).where(WorkItem.job_id == job_id, WorkItem.status == "running").values(
            lease_until=time.time() - 1,
        ))

    def drain(worker):
        while worker.run_once():
            pass

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(drain, workers))
    with database.session() as session:
        job = session.get(Job, job_id)
        assert job.status == "completed"
        items = list(session.scalars(select(WorkItem).where(WorkItem.job_id == job_id)))
        assert len(items) == 6 and all(item.status == "completed" for item in items)
        annotations = list(session.scalars(select(Annotation).where(Annotation.job_id == job_id)))
        assert len(annotations) > 6
        assert len({(a.record_id, a.target_id) for a in annotations}) == len(annotations)
