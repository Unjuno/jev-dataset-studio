# Internal interfaces


Package: `studio`. Core files are owned by Task 1. Root owns persistence/API/worker/exports and packaging. Task 2 owns `studio/static/` only.

### Typed recipe JSON

```json
{
  "version": 1,
  "name": "Equipment annotations",
  "provider": "demo",
  "model": "jev-1.14.0",
  "source": {"text_field": "text", "id_field": "id", "group_field": null},
  "target": {"mode": "sentence", "terms": [], "pattern": null, "ignore_case": false, "max_candidates": 200},
  "context": {"mode": "window", "window_chars": 180, "max_chars": 12000},
  "features": [
    {"id": "category", "name": "分類", "kind": "choice", "instruction": "Classify the target phenomenon.", "choices": {"vibration": "Vibration / 振動", "heat": "Heat / 発熱", "other": "Other or unknown"}, "levels": [], "filter": false, "minimum": null},
    {"id": "important", "name": "重要箇所", "kind": "boolean", "instruction": "Does this passage affect an inspection decision?", "choices": {}, "levels": [], "filter": false, "minimum": null}
  ],
  "quality": {"min_confidence": 0.6, "boolean_low": 0.2, "boolean_high": 0.8, "secondary_context": null, "verifier": false, "score_tolerance": 0.5, "audit_fraction": 0.05}
}
```

Target modes: `document`, `sentence`, `paragraph`, `literal`, `regex`. Context modes: `target`, `sentence`, `window`, `paragraph`, `document`. Feature kinds: `choice`, `multi_label`, `boolean`, `score`. `choices` is a bounded ordered string-to-description map. Score `levels` are 2–10 descriptive levels. `filter` is allowed only for boolean; `minimum` only for score. Every feature requires a meaningful instruction; semantic ambiguity is not solved by its ID. Text must never be silently truncated; oversize context or excessive candidates produces a clear error.

### Core Python APIs (Task 1 must implement these)

- `studio.schemas`: Pydantic models `Recipe`, `SourceSpec`, `TargetSpec`, `ContextSpec`, `FeatureSpec`, `QualitySpec`, `Target` (`id`, `start`, `end`, `text`). All recipe models forbid unknown keys and non-finite numbers. Stable hash helper `canonical_hash(value) -> str`.
- `studio.targets.generate_targets(text: str, spec: TargetSpec) -> list[Target]`; `build_context(text: str, target: Target, spec: ContextSpec) -> dict`. IDs deterministically distinguish repeated occurrences. Use `regex` timeout for user regex; match no zero-length span. Return original codepoint offsets; do not normalize text.
- `studio.providers.ProviderResult`: dataclass with `answers: dict[str, dict]`, `model: str`, `usage: dict`, `origin: str`. Each answer has `value`, `probabilities` (dict), `confidence` (float or None). For boolean also preserve `probability` (yes). For multi_label preserve per-label yes probabilities; value is list or None. Primary decoding/abstention is finalized by engine thresholds, not by pretending confidence=accuracy.
- `studio.providers.ProviderError(message: str, retryable: bool=False)`; `DemoProvider(model="demo-v1")`, `JevProvider(api_key: str, model: str, *, transport=None)`, `VerifierProvider(api_key: str, base_url: str, model: str, *, transport=None)`. Providers expose `.identity` string and `.evaluate(state: dict, features: list[FeatureSpec]) -> ProviderResult`. One HTTP attempt per call, worker handles bounded retry; no provider-internal invisible calls. Jev URL fixed to `https://api.typesafe.ai/v1/systemone`. Verifier URL is trusted administrator configuration, HTTPS, no redirects, OpenAI-compatible chat/completions; parse strict structured answer JSON using same feature definitions and include unknown values explicitly. Do not reveal provider response bodies containing secrets in exceptions. HTTP invalid output is error. Validate all feature IDs, choices, ranges, finite probabilities; reject incomplete or fabricated keys. Demo always `origin="demo"`; live Jev `origin="model"`; verifier `origin="verifier"`.
- `studio.engine.annotate_target(text: str, target: Target, recipe: Recipe, provider, verifier=None) -> dict`. Return `target` dict, `features` dict, `predictions` list, `status` (`accepted`, `review`, `excluded`), `reasons` list, `origin`, `reviewed` false. Predictions record stage, actual context, model, origin, answers, usage. Compare primary and secondary contexts for the same features; second provider does not see first provider's answers. All target/feature alignment checked. Demo always review. Ambiguous boolean/multi_label probabilities -> None or explicit uncertainty plus review, never a forced negative. Filter-negative may be excluded only when confidently classified. Low confidence/disagreement/audit -> review. Audit selection deterministic by target/text/recipe hash. No external storage in core.

### HTTP contract (root implements)

All JSON endpoints are under `/api`. JSON errors: `{"detail": ...}`; never swallow errors. Every POST/PATCH request (including multipart upload, login, and logout) must set `X-Studio-Request: 1`. Use same-origin cookies; no credentials in localStorage. `POST /api/jobs/{id}/resume` optionally accepts `{max_requests,budget_usd}` to raise an exhausted limit. Other job actions can have no body.

- `GET /api/session` -> `{authenticated, auth_required, workspace, provider_ready:{jev,verifier}, default_model, limits}`.
- `POST /api/login` `{token}`; `POST /api/logout`.
- `GET /api/datasets` -> `[{id,name,rows,columns,created_at}]`.
- `POST /api/datasets` multipart `file` -> dataset summary plus `sample` list (up to 10 rows). CSV, JSON/JSONL, TXT and Parquet; bounded size/rows. IDs/columns from data inferred but never silently lose strings or nested data.
- `POST /api/datasets/sample` -> built-in synthetic Japanese sample dataset summary/sample.
- `GET /api/datasets/{id}` -> summary plus sample; `GET /api/datasets/{id}/records?offset=0&limit=20` -> `{items,total}`.
- `GET /api/templates` -> `[{id,name,recipe,export_spec}]`; `POST /api/templates` `{name,recipe,export_spec}` -> saved version/id. Built-in templates equipment/generic supplied, no raw private data in templates.
- `POST /api/preview` `{dataset_id,recipe,limit:3,max_requests:100}` -> `{items:[annotation...],stats:{...},errors:[],request_count,...}`. Preview uses same processing code as worker; each annotation includes `id,source_id,record_id,text,target,features,status,reasons,origin,reviewed,predictions`.
- `POST /api/jobs` `{dataset_id,recipe,max_requests:10000,budget_usd:null}` -> job.
- `GET /api/jobs` -> list; `GET /api/jobs/{id}` -> `{id,status,total,completed,failed,request_count,estimated_cost_usd,actual_input_tokens,errors,recipe,dataset_id,created_at}`.
- `POST /api/jobs/{id}/pause`, `/resume`, `/cancel`, `/retry-failed` -> job.
- `GET /api/jobs/{id}/annotations?status=all&offset=0&limit=50` -> `{items,total,counts}`. Each annotation includes source text, target, features, reasons, provenance, revision and predictions.
- `PATCH /api/annotations/{id}` `{revision,decision:"approved"|"rejected",features:{...},note:""}` -> updated annotation. Validate edited features against recipe; optimistic revision check; store review separately.
- `POST /api/jobs/{id}/exports` `{format:"jsonl"|"csv"|"parquet",grain:"target"|"document"|"annotation",selection:"accepted"|"reviewed"|"all",columns:[{name,source,role:"input"|"target"|"metadata"|"group"}],split:{enabled:false,train:0.8,validation:0.1,test:0.1,seed:"42"},include_empty:false}` -> `{id,download_url,rows,manifest}`. Sources allow `source_id`, `text`, `target.text`, `target.start`, `target.end`, `features.<id>`, `status`, `origin`, `reviewed`, `split`, `feature_id`, `value`, `source.<column>`. Document grain always has arrays for annotation-level values, including singleton and empty arrays. CSV uses explicit JSON serialization. Export ZIP holds dataset file(s), provenance.jsonl, reviews.jsonl, manifest.json, recipe.json. Review state never dropped from provenance. `include_empty` is available only in document grain. Approved reviews cannot retain null feature values. Demo rows remain `origin=demo` even after human review.
- `GET /api/exports/{id}/download` -> zip file (authenticated).
- `GET /api/health` -> `{status:"ok",database:"ok"}`.
