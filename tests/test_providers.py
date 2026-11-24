import json

import httpx
import pytest

from studio.engine import annotate_target
from studio.providers import DemoProvider, JevProvider, ProviderError, VerifierProvider
from studio.schemas import FeatureSpec, QualitySpec, Recipe, Target


FEATURES = [
    FeatureSpec(id="kind", name="Kind", kind="choice", instruction="Classify the described fault.",
                choices={"heat": "Overheating", "other": "Other or unknown"}),
    FeatureSpec(id="signs", name="Signs", kind="multi_label", instruction="Identify which signs occur.",
                choices={"noise": "Abnormal noise", "heat": "Unusual heat"}),
    FeatureSpec(id="report", name="Report", kind="boolean", instruction="Was an incident reported?"),
    FeatureSpec(id="severity", name="Severity", kind="score", instruction="Rate the response urgency.",
                levels=["Routine", "Urgent", "Emergency"]),
]
STATE = {"text": "Pump overheated.", "target": {"text": "Pump"}}


def response_body(questions):
    by_type = {
        "choice": {"type": "choice", "choice": "heat", "confidence": 0.8,
                   "probabilities": {"heat": 0.9, "other": 0.1}},
        "score": {"type": "score", "score": 1.48, "confidence": 0.28,
                  "legend": {"0": "Routine", "1": "Urgent", "2": "Emergency"},
                  "probabilities": {"0": 0, "1": 0.52, "2": 0.48}},
    }
    return {
        "model": "jev-1.14.0",
        "usage": {"input_tokens": 341, "output_tokens": 20},
        "answers": {
            uid: by_type.get(
                question["type"], {"type": "noul", "noul": 0.95 if uid == "report" else 0.15},
            )
            for uid, question in questions.items()
        },
    }


def make_jev(handler):
    return JevProvider("fixture-token", "jev-latest", transport=httpx.MockTransport(handler))


def test_jev_sends_all_features_in_one_call_and_preserves_probability_shapes():
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append(request)
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.method == "POST"
        assert request.headers["Authorization"] == "Bearer fixture-token"
        assert payload["state"] == STATE
        assert payload["model"] == "jev-latest"
        questions = payload["questions"]
        assert questions["kind"] == {
            "type": "choice", "instructions": "Classify the described fault.",
            "criteria": {"heat": "Overheating", "other": "Other or unknown"},
        }
        assert questions["report"]["type"] == "noul"
        assert questions["severity"]["criteria"] == ["Routine", "Urgent", "Emergency"]
        label_questions = [q for uid, q in questions.items() if uid not in {"kind", "report", "severity"}]
        assert len(label_questions) == 2
        assert all(q["type"] == "noul" for q in label_questions)
        assert "Abnormal noise" in json.dumps(label_questions)
        assert "Unusual heat" in json.dumps(label_questions)
        return httpx.Response(200, json=response_body(questions))

    result = make_jev(handle).evaluate(STATE, FEATURES)
    assert len(calls) == 1
    assert result.origin == "model"
    assert result.model == "jev-1.14.0"
    assert result.answers["kind"]["value"] == "heat"
    assert result.answers["report"]["probability"] == 0.95
    assert result.answers["report"]["confidence"] is None
    assert result.answers["severity"]["value"] == 1.48
    assert result.answers["signs"]["probabilities"] == {"noise": 0.15, "heat": 0.15}
    assert result.usage == {"input_tokens": 341, "output_tokens": 20}


@pytest.mark.parametrize("break_response", [
    lambda body: body["answers"].pop("report"),
    lambda body: body["answers"].update(unknown={"type": "noul", "noul": 0}),
    lambda body: body["answers"]["kind"].update(choice="invented"),
    lambda body: body["answers"]["kind"]["probabilities"].pop("other"),
    lambda body: body["answers"]["kind"]["probabilities"].update(other=0.7),
    lambda body: body["answers"]["kind"].update(choice="other"),
    lambda body: body["answers"]["report"].update(noul=True),
    lambda body: body["answers"]["report"].update(noul=1.01),
    lambda body: body["answers"]["severity"].update(score=3),
    lambda body: body["answers"]["severity"].update(score=0.25),
    lambda body: body["answers"]["severity"]["legend"].update({"2": "Invented"}),
    lambda body: body["usage"].update(input_tokens=-1),
    lambda body: body.pop("model"),
])
def test_jev_errors_on_incomplete_mismatched_or_invalid_answers(break_response):
    def handle(request):
        body = response_body(json.loads(request.content)["questions"])
        break_response(body)
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderError) as exc:
        make_jev(handle).evaluate(STATE, FEATURES)
    assert not exc.value.retryable


@pytest.mark.parametrize("body", [
    '{"model":"jev","answers":{"report":{"type":"noul","noul":NaN}},"usage":{"input_tokens":1,"output_tokens":1}}',
    '{"model":"jev","answers":{"report":{"type":"noul","noul":0,"noul":1}},"usage":{"input_tokens":1,"output_tokens":1}}',
    '{"model":"jev","answers":',
])
def test_jev_rejects_malformed_json_even_if_parser_can_salvage_it(body):
    provider = make_jev(lambda request: httpx.Response(200, text=body))
    with pytest.raises(ProviderError):
        provider.evaluate(STATE, FEATURES[2:3])


@pytest.mark.parametrize("status, retryable", [(400, False), (401, False), (307, False), (429, True), (500, True)])
def test_jev_http_errors_are_redacted_and_never_retried_in_provider(status, retryable, caplog):
    calls = []
    secret_marker = "synthetic-response-secret-marker"

    def handle(request):
        calls.append(request)
        return httpx.Response(status, headers={"Location": "https://example.com"}, text=secret_marker)

    with pytest.raises(ProviderError) as exc:
        make_jev(handle).evaluate(STATE, FEATURES)
    assert exc.value.retryable is retryable
    assert len(calls) == 1
    assert secret_marker not in str(exc.value)
    assert secret_marker not in caplog.text
    assert "fixture-token" not in str(exc.value)
    assert "fixture-token" not in caplog.text


def test_jev_http_timeout_is_retryable_and_redacted():
    def handle(request):
        raise httpx.ReadTimeout("synthetic-transport-secret-marker", request=request)

    with pytest.raises(ProviderError) as exc:
        make_jev(handle).evaluate(STATE, FEATURES)
    assert exc.value.retryable
    assert "synthetic-transport-secret-marker" not in str(exc.value)


def make_verifier(handler):
    return VerifierProvider(
        "fixture-token", "https://verifier.invalid/v1", "fixture-model",
        transport=httpx.MockTransport(handler),
    )


def verifier_response(content, *, finish_reason="stop", refusal=None, model="fixture-model-actual"):
    return httpx.Response(200, json={
        "model": model,
        "choices": [{"message": {"content": content, "refusal": refusal}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 38, "completion_tokens": 14, "total_tokens": 52},
    })


@pytest.mark.parametrize("provider_name", ["jev", "verifier"])
@pytest.mark.parametrize("api_key", ["@", "mockkey"])
def test_short_credential_is_rejected_before_prediction_provenance(provider_name, api_key):
    model = f"fixture-{api_key}-response"

    def handle(request):
        if provider_name == "jev":
            body = response_body(json.loads(request.content)["questions"])
            return httpx.Response(200, json={**body, "model": model})
        return verifier_response('{"answers":{"report":false}}', model=model)

    transport = httpx.MockTransport(handle)
    if provider_name == "jev":
        primary = JevProvider(api_key, "jev-latest", transport=transport)
        verifier = None
    else:
        primary = DemoProvider()
        verifier = VerifierProvider(api_key, "https://verifier.invalid/v1", "fixture-model", transport=transport)
    recipe = Recipe(features=FEATURES[2:3], quality=QualitySpec(verifier=verifier is not None))
    target = Target(id="t", start=0, end=4, text="Pump")

    with pytest.raises(ProviderError, match="invalid prediction data") as exc:
        annotate_target("Pump overheated.", target, recipe, primary, verifier)
    assert api_key not in str(exc.value)


def test_verifier_returns_explicit_unknown_and_validates_feature_specific_types():
    def handle(request):
        payload = json.loads(request.content)
        assert str(request.url) == "https://verifier.invalid/v1/chat/completions"
        assert [msg["role"] for msg in payload["messages"]] == ["system", "user"]
        assert json.loads(payload["messages"][1]["content"]) == STATE
        schema = payload["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]["answers"]["required"]) == {"kind", "signs", "report", "severity"}
        return verifier_response(
            json.dumps({"answers": {"kind": None, "signs": ["heat"], "report": False, "severity": 1.7}}),
        )

    result = make_verifier(handle).evaluate(STATE, FEATURES)
    assert result.origin == "verifier"
    assert result.model == "fixture-model-actual"
    assert result.answers["kind"]["value"] is None
    assert result.answers["signs"]["value"] == ["heat"]
    assert result.answers["report"]["value"] is False
    assert result.answers["report"]["probability"] is None
    assert result.answers["severity"]["value"] == 1.7
    assert result.usage == {"input_tokens": 38, "output_tokens": 14}


@pytest.mark.parametrize("answers", [
    {"kind": None, "signs": [], "report": False},
    {"kind": "new", "signs": [], "report": False, "severity": 0},
    {"kind": "heat", "signs": ["heat", "heat"], "report": False, "severity": 0},
    {"kind": "heat", "signs": ["water"], "report": False, "severity": 0},
    {"kind": "heat", "signs": [], "report": 0, "severity": 0},
    {"kind": "heat", "signs": [], "report": False, "severity": True},
    {"kind": "heat", "signs": [], "report": False, "severity": -1},
    {"kind": "heat", "signs": [], "report": False, "severity": 0, "extra": None},
])
def test_verifier_rejects_missing_duplicate_fabricated_or_wrong_type_values(answers):
    provider = make_verifier(lambda request: verifier_response(json.dumps({"answers": answers})))
    with pytest.raises(ProviderError):
        provider.evaluate(STATE, FEATURES)


@pytest.mark.parametrize("content, finish, refusal", [
    ('{"answers":{"report":false}}', "length", None),
    ('{"answers":{"report":false}}', "stop", "SYNTHETIC"),
    ('```json\n{"answers":{"report":false}}\n```', "stop", None),
    ('{"answers":{"report":false},"extra":true}', "stop", None),
])
def test_verifier_rejects_incomplete_refused_or_non_json_answers(content, finish, refusal):
    provider = make_verifier(lambda request: verifier_response(content, finish_reason=finish, refusal=refusal))
    with pytest.raises(ProviderError):
        provider.evaluate(STATE, FEATURES[2:3])


@pytest.mark.parametrize("url", [
    "http://verifier.invalid/v1",
    "https://secret@verifier.invalid/v1",
    "https://verifier.invalid/v1?token=secret",
    "https://verifier.invalid/v1#fragment",
])
def test_verifier_rejects_unusable_base_urls_without_echoing_them(url):
    with pytest.raises(ValueError) as exc:
        VerifierProvider("fixture-token", url, "fixture-model")
    assert url not in str(exc.value)
