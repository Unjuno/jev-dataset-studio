"""The same inference path for preview and durable jobs; no hidden requests."""

import json
import re
import time

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from .db import Job, PredictionCache, WorkItem
from .engine import annotate_target
from .ingest import source_id
from .providers import DemoProvider, JevProvider, ProviderError, ProviderResult, VerifierProvider
from .schemas import canonical_hash
from .targets import generate_targets


class JobInterrupted(Exception):
    """A stopped job or expired lease must not publish another result."""


class BudgetExceeded(Exception):
    pass


def ensure_providers(recipe, settings):
    if recipe.provider == "jev" and not settings.jev_api_key:
        raise ValueError("Jev APIキーがサーバーに設定されていません / Configure TYPESAFE_API_KEY.")
    if recipe.quality.verifier and not settings.verifier_ready:
        raise ValueError("別モデルの検証にはサーバー側のVERIFIER設定が必要です / Configure the verifier.")


def input_tokens(usage):
    for name in ("input_tokens", "prompt_tokens", "total_input_tokens"):
        value = usage.get(name)
        if type(value) is int and value >= 0:
            return value
    return 0


class RequestBudget:
    def __init__(self, database, settings, max_requests=100, *, job_id=None, item_id=None, lease=None):
        self.database, self.settings = database, settings
        self.max_requests = max_requests
        self.job_id, self.item_id, self.lease = job_id, item_id, lease
        self.request_count = 0
        self.cache_hits = 0
        self.actual_input_tokens = 0
        self.estimated_cost_usd = 0.0

    def _locked_job(self, session):
        job = session.scalar(select(Job).where(Job.id == self.job_id).with_for_update())
        if not job or job.status not in {"queued", "running"}:
            raise JobInterrupted("Job is paused or stopped.")
        item = session.scalar(select(WorkItem).where(WorkItem.id == self.item_id).with_for_update())
        if not item or item.status != "running" or item.lease_token != self.lease or item.lease_until < time.time():
            raise JobInterrupted("Worker lease expired; the current worker cannot publish results.")
        item.lease_until = time.time() + self.settings.lease_seconds
        return job

    def check(self):
        if self.job_id:
            with self.database.session() as session:
                self._locked_job(session)

    def reserve(self, state, features, provider):
        # A UTF-8 byte estimate is deliberately conservative for typical text, but
        # is not a tokenizer or a provider invoice. The request cap is exact.
        payload = {"state": state, "features": [f.model_dump(mode="json") for f in features]}
        estimated_tokens = len(json.dumps(payload, ensure_ascii=False).encode()) + 512
        is_demo = isinstance(provider, DemoProvider)
        price = (self.settings.verifier_price_per_million if isinstance(provider, VerifierProvider)
                 else self.settings.input_price_per_million)
        estimated_cost = 0.0 if is_demo else estimated_tokens * price / 1_000_000
        if self.job_id:
            with self.database.session() as session:
                job = self._locked_job(session)
                # Conditional increments avoid lost budget reservations on SQLite.
                conditions = [Job.id == job.id, Job.status.in_(["queued", "running"]),
                              Job.request_count < Job.max_requests]
                if job.budget_usd is not None:
                    conditions.append(Job.estimated_cost_usd + estimated_cost <= Job.budget_usd)
                result = session.execute(update(Job).where(*conditions).values(
                    request_count=Job.request_count + 1,
                    estimated_cost_usd=Job.estimated_cost_usd + estimated_cost,
                    updated_at=time.time(),
                ), execution_options={"synchronize_session": False})
                if result.rowcount != 1:
                    raise BudgetExceeded("Request or estimated-cost limit reached. Increase the limits to resume.")
        elif self.request_count >= self.max_requests:
            raise BudgetExceeded("Preview request limit reached. Reduce the sample, targets, or checks.")
        self.request_count += 1
        self.estimated_cost_usd += estimated_cost

    def record_usage(self, usage):
        tokens = input_tokens(usage)
        self.actual_input_tokens += tokens
        if self.job_id and tokens:
            with self.database.session() as session:
                session.execute(update(Job).where(Job.id == self.job_id).values(
                    actual_input_tokens=Job.actual_input_tokens + tokens,
                ))


class CachedProvider:
    """Cache raw feature predictions, never review/acceptance decisions."""

    def __init__(self, provider, database, owner, budget):
        self.provider, self.database, self.owner, self.budget = provider, database, owner, budget
        self.identity = provider.identity
        # Movable aliases and arbitrary verifier services have no immutable model
        # revision guarantee. Their calls always run rather than reuse stale labels.
        self.cacheable = isinstance(provider, DemoProvider) or (
            isinstance(provider, JevProvider) and re.fullmatch(r"jev-\d+\.\d+\.\d+", provider.model) is not None
        )

    def evaluate(self, state, features):
        self.budget.check()
        keys = {f.id: canonical_hash({
            "cache_version": 1, "provider": self.identity, "state": state,
            "feature": canonical_hash(f),
        }) for f in features}
        with self.database.session() as session:
            rows = list(session.scalars(select(PredictionCache).where(
                PredictionCache.owner == self.owner, PredictionCache.cache_key.in_(keys.values()),
            )))
        existing = {r.cache_key: r.result for r in rows
                    if self.cacheable and r.result["model"] == self.provider.model}
        answers = {}
        missing = []
        model, origin = None, None
        for feature in features:
            cached = existing.get(keys[feature.id])
            if cached:
                answers[feature.id] = cached["answer"]
                model, origin = cached["model"], cached["origin"]
                self.budget.cache_hits += 1
            else:
                missing.append(feature)
        usage = {"input_tokens": 0, "cached_features": len(features) - len(missing)}
        if missing:
            self.budget.reserve(state, missing, self.provider)
            result = self.provider.evaluate(state, missing)
            self.budget.record_usage(result.usage)
            if answers and result.model != model:
                raise ProviderError("The resolved model changed within a cached prediction. Use a pinned model version.")
            answers.update(result.answers)
            model, origin = result.model, result.origin
            usage = {**result.usage, "cached_features": len(features) - len(missing)}
            for feature in missing:
                if not self.cacheable or result.model != self.provider.model:
                    continue
                try:
                    with self.database.session() as session:
                        session.add(PredictionCache(owner=self.owner, cache_key=keys[feature.id], result={
                            "answer": result.answers[feature.id], "model": model, "origin": origin,
                            "original_usage": result.usage,
                        }))
                except IntegrityError:
                    # Two workers may compute the same value. Unique keys keep one
                    # cache entry; both actual requests remain in their own budgets.
                    pass
        return ProviderResult(answers=answers, model=model, usage=usage, origin=origin)


def make_providers(recipe, settings, database, owner, budget):
    ensure_providers(recipe, settings)
    primary = (DemoProvider() if recipe.provider == "demo" else
               JevProvider(settings.jev_api_key, recipe.model))
    verifier = (VerifierProvider(settings.verifier_api_key, settings.verifier_base_url, settings.verifier_model)
                if recipe.quality.verifier else None)
    return (CachedProvider(primary, database, owner, budget),
            CachedProvider(verifier, database, owner, budget) if verifier else None)


def process_record(record, recipe, provider, verifier=None):
    text = record.values[recipe.source.text_field]
    annotations = []
    for target in generate_targets(text, recipe.target):
        annotation = annotate_target(text, target, recipe, provider, verifier)
        annotation.update({"source_id": source_id(record, recipe), "record_id": record.id, "text": text})
        annotations.append(annotation)
    return annotations
