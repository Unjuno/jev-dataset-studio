"""Durable SQL work items with leases, fenced writes, and bounded retries."""

import logging
import time
from threading import Event

from sqlalchemy import and_, func, or_, select, update

from .db import Annotation, Job, Record, WorkItem, uid
from .providers import ProviderError
from .runtime import BudgetExceeded, JobInterrupted, RequestBudget, make_providers, process_record
from .schemas import Recipe

logger = logging.getLogger(__name__)


def finish_job(database, job_id):
    with database.session() as session:
        counts = dict(session.execute(select(WorkItem.status, func.count()).where(
            WorkItem.job_id == job_id,
        ).group_by(WorkItem.status)).all())
        if not counts.get("pending") and not counts.get("running"):
            session.execute(update(Job).where(Job.id == job_id, Job.status.in_(["queued", "running"])).values(
                status="completed_with_errors" if counts.get("failed") else "completed",
                updated_at=time.time(),
            ))


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

    def _item(self, session, item_id, job_id, token, *, allow_stopped=False):
        job = session.scalar(select(Job).where(Job.id == job_id).with_for_update())
        item = session.scalar(select(WorkItem).where(WorkItem.id == item_id).with_for_update())
        if not item or item.lease_token != token or item.status != "running" or item.lease_until < time.time():
            raise JobInterrupted("Worker lease no longer belongs to this attempt.")
        if not allow_stopped and job.status not in {"running", "queued"}:
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
        except BudgetExceeded as error:
            try:
                with self.database.session() as session:
                    job, item = self._item(session, item_id, job_id, token, allow_stopped=True)
                    if job.status in {"queued", "running"}:
                        job.status, job.message = "paused", str(error)
                    item.status, item.lease_token, item.lease_until = "pending", None, 0
                    item.attempts = max(0, item.attempts - 1)
            except JobInterrupted:
                pass
        except JobInterrupted:
            # A paused attempt can release its own lease; a stale attempt cannot.
            with self.database.session() as session:
                session.execute(update(WorkItem).where(
                    WorkItem.id == item_id, WorkItem.status == "running", WorkItem.lease_token == token,
                    WorkItem.lease_until >= time.time(),
                ).values(status="pending", lease_token=None, lease_until=0,
                         attempts=WorkItem.attempts - 1))
        except Exception as error:
            retryable = isinstance(error, ProviderError) and error.retryable
            safe_message = str(error) if isinstance(error, (ProviderError, ValueError)) else "Internal processing error; inspect server logs using the job ID."
            if not isinstance(error, (ProviderError, ValueError)):
                # Never log source text, credentials, or provider response bodies.
                logger.error("Job %s work item %s failed (%s)", job_id, item_id, type(error).__name__)
            try:
                with self.database.session() as session:
                    _job, item = self._item(session, item_id, job_id, token, allow_stopped=True)
                    item.status = "pending" if retryable and item.attempts < self.settings.max_attempts else "failed"
                    item.available_at = time.time() + min(30, 2 ** item.attempts)
                    item.error = safe_message[:1000]
                    item.lease_token, item.lease_until = None, 0
            except JobInterrupted:
                pass
        finish_job(self.database, job_id)
        return True

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
