"""Durable SQL work items with leases, fenced writes, and bounded retries."""

import logging
import time
from threading import Event

from sqlalchemy import and_, or_, select, update

from .db import Annotation, Job, Record, WorkItem, uid
from .providers import ProviderError
from .runtime import BudgetExceeded, JobInterrupted, RequestBudget, make_providers, process_record
from .schemas import Recipe

logger = logging.getLogger(__name__)


def finish_job(session, job):
    """Finalize under the caller's job lock, in the same commit as the last item."""
    if job.status not in {"queued", "running"}:
        return
    items = select(WorkItem.id).where(WorkItem.job_id == job.id)
    if session.scalar(select(items.where(WorkItem.status.in_(["pending", "running"])).exists())):
        return
    failed = session.scalar(select(items.where(WorkItem.status == "failed").exists()))
    job.status = "completed_with_errors" if failed else "completed"
    job.updated_at = time.time()


def stop_job(session, job, status):
    """Fence in-flight successes and failures even if the job resumes immediately."""
    job.status, job.updated_at = status, time.time()
    session.execute(update(WorkItem).where(
        WorkItem.job_id == job.id, WorkItem.status == "running",
    ).values(status="pending", lease_token=None, lease_until=0, attempts=WorkItem.attempts - 1))


class Worker:
    def __init__(self, database, settings, provider_factory=make_providers):
        self.database, self.settings, self.provider_factory = database, settings, provider_factory

    def claim(self):
        now = time.time()
        ready = or_(and_(WorkItem.status == "pending", WorkItem.available_at <= now),
                    and_(WorkItem.status == "running", WorkItem.lease_until < now))
        with self.database.session() as session:
            query = select(WorkItem).join(Job, WorkItem.job_id == Job.id).where(
                Job.status.in_(["queued", "running"]), ready,
            ).order_by(Job.created_at, WorkItem.id).limit(1)
            if self.database.engine.dialect.name == "postgresql":
                query = query.with_for_update(skip_locked=True, of=WorkItem)
            item = session.scalar(query)
            if not item:
                return None
            token = uid()
            result = session.execute(update(WorkItem).where(WorkItem.id == item.id, ready).values(
                status="running", lease_token=token, lease_until=now + self.settings.lease_seconds,
                attempts=WorkItem.attempts + 1,
            ), execution_options={"synchronize_session": False})
            if result.rowcount != 1:
                return None
            claim = (item.id, item.job_id, token)
        with self.database.session() as session:
            session.execute(update(Job).where(Job.id == claim[1], Job.status == "queued").values(
                status="running", updated_at=time.time(),
            ))
        return claim

    def _item(self, session, item_id, job_id, token):
        job = session.scalar(select(Job).where(Job.id == job_id).with_for_update())
        item = session.scalar(select(WorkItem).where(WorkItem.id == item_id).with_for_update())
        if not item or item.lease_token != token or item.status != "running" or item.lease_until < time.time():
            raise JobInterrupted("Worker lease no longer belongs to this attempt.")
        if job.status not in {"running", "queued"}:
            raise JobInterrupted("Job stopped while this record was running.")
        return job, item

    def run_once(self):
        claim = self.claim()
        if not claim:
            return False
        item_id, job_id, token = claim
        try:
            with self.database.session() as session:
                job, item = self._item(session, item_id, job_id, token)
                if item.attempts > self.settings.max_attempts:
                    raise ProviderError("Worker retry limit exceeded after interrupted attempts.")
                record = session.get(Record, item.record_id)
                recipe = Recipe.model_validate(job.recipe)
                owner = job.owner
            budget = RequestBudget(self.database, self.settings, job_id=job_id, item_id=item_id, lease=token)
            primary, verifier = self.provider_factory(recipe, self.settings, self.database, owner, budget)
            annotations = process_record(record, recipe, primary, verifier)
            with self.database.session() as session:
                job, item = self._item(session, item_id, job_id, token)
                for annotation in annotations:
                    session.add(Annotation(
                        owner=owner, job_id=job_id, record_id=record.id,
                        source_id=annotation["source_id"], target_id=annotation["target"]["id"],
                        target=annotation["target"], features=annotation["features"],
                        original_features=annotation["features"], predictions=annotation["predictions"],
                        reasons=annotation["reasons"], status=annotation["status"], origin=annotation["origin"],
                        reviewed=False,
                    ))
                item.status, item.lease_token, item.lease_until = "completed", None, 0
                item.error = ""
                item.result = {"annotations": len(annotations), "cache_hits": budget.cache_hits,
                               "no_targets": not annotations}
                finish_job(session, job)
        except BudgetExceeded as error:
            try:
                with self.database.session() as session:
                    job, _ = self._item(session, item_id, job_id, token)
                    stop_job(session, job, "paused")
                    job.message = str(error)
            except JobInterrupted:
                self._release_interrupted(item_id, token)
        except JobInterrupted:
            self._release_interrupted(item_id, token)
        except Exception as error:
            retryable = isinstance(error, ProviderError) and error.retryable
            safe_message = str(error) if isinstance(error, (ProviderError, ValueError)) else "Internal processing error; inspect server logs using the job ID."
            if not isinstance(error, (ProviderError, ValueError)):
                # Never log source text, credentials, or provider response bodies.
                logger.error("Job %s work item %s failed (%s)", job_id, item_id, type(error).__name__)
            try:
                with self.database.session() as session:
                    job, item = self._item(session, item_id, job_id, token)
                    item.status = "pending" if retryable and item.attempts < self.settings.max_attempts else "failed"
                    if item.status == "pending":
                        item.available_at = time.time() + min(30, 2 ** item.attempts)
                    item.error = safe_message[:1000]
                    item.lease_token, item.lease_until = None, 0
                    if item.status == "failed":
                        finish_job(session, job)
            except JobInterrupted:
                self._release_interrupted(item_id, token)
        return True

    def _release_interrupted(self, item_id, token):
        # A claim racing with a stop may release its own lease, never a replacement.
        with self.database.session() as session:
            session.execute(update(WorkItem).where(
                WorkItem.id == item_id, WorkItem.status == "running", WorkItem.lease_token == token,
                WorkItem.lease_until >= time.time(),
            ).values(status="pending", lease_token=None, lease_until=0, attempts=WorkItem.attempts - 1))

    def run_forever(self, stop: Event | None = None):
        stop = stop or Event()
        while not stop.is_set():
            try:
                worked = self.run_once()
            except Exception as error:
                logger.error("Worker database operation failed (%s)", type(error).__name__)
                worked = False
            if not worked:
                stop.wait(1)
