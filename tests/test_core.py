from copy import deepcopy

import pytest
import regex
from pydantic import ValidationError

from studio.engine import annotate_target
from studio.providers import DemoProvider, ProviderError, ProviderResult
from studio.schemas import ContextSpec, FeatureSpec, QualitySpec, Recipe, Target, TargetSpec, canonical_hash
from studio.targets import build_context, generate_targets


def make_recipe(**quality):
    return Recipe(
        name="Inspection",
        provider="jev",
        context=ContextSpec(mode="target"),
        features=[
            FeatureSpec(
                id="relevant",
                name="Relevant",
                kind="boolean",
                instruction="Does the passage affect the inspection decision?",
                filter=True,
            ),
        ],
        quality=QualitySpec(audit_fraction=quality.pop("audit_fraction", 0), **quality),
    )


@pytest.mark.parametrize("spec, extra", [
    (TargetSpec, {"unknown": 1}),
    (TargetSpec, {"mode": "regex", "pattern": "["}),
    (TargetSpec, {"mode": "literal", "terms": []}),
    (ContextSpec, {"max_chars": 0}),
    (QualitySpec, {"audit_fraction": float("nan")}),
    (QualitySpec, {"boolean_low": 0.6, "boolean_high": 0.7}),
    (QualitySpec, {"min_confidence": True}),
])
def test_specs_reject_invalid_settings(spec, extra):
    with pytest.raises(ValidationError):
        spec(**extra)


@pytest.mark.parametrize("kind, extra", [
    ("choice", {"choices": {"x": "X only"}}),
    ("multi_label", {"choices": {"x": " "}}),
    ("score", {"levels": ["low"]}),
    ("score", {"levels": ["low", "high"], "minimum": 2}),
    ("score", {"levels": ["low", "high"], "filter": True}),
    ("boolean", {"minimum": 0}),
    ("boolean", {"instruction": "    "}),
])
def test_feature_rejects_invalid_kind_specific_configuration(kind, extra):
    info = {"instruction": "Judge the relevance of the evidence.", **extra}
    with pytest.raises(ValidationError):
        FeatureSpec(id="f", name="Feature", kind=kind, **info)


def test_recipe_rejects_duplicate_feature_ids():
    feature = make_recipe().features[0]
    with pytest.raises(ValidationError):
        Recipe(name="Inspection", features=[feature, feature])


@pytest.mark.parametrize("version", [True, 1.0])
def test_recipe_version_requires_a_json_integer(version):
    with pytest.raises(ValidationError):
        Recipe(version=version, features=make_recipe().features)


def test_hash_preserves_choice_order_and_canonicalizes_unordered_objects():
    first = FeatureSpec(id="f", name="Feature", kind="choice", instruction="Classify the fault.",
                        choices={"heat": "Too hot", "power": "No power"})
    second = FeatureSpec(id="f", name="Feature", kind="choice", instruction="Classify the fault.",
                         choices={"power": "No power", "heat": "Too hot"})
    assert canonical_hash(Recipe(name="Inspection", features=[first])) != canonical_hash(
        Recipe(name="Inspection", features=[second])
    )
    assert canonical_hash({"nested": {"a": 1, "b": 2}, "c": None}) == canonical_hash(
        {"c": None, "nested": {"b": 2, "a": 1}}
    )


def test_literal_targets_preserve_original_codepoint_offsets():
    text = "😀é 😀é é"
    spec = TargetSpec(mode="literal", terms=["😀é"])
    targets = generate_targets(text, spec)
    assert [(t.start, t.end, t.text) for t in targets] == [(0, 3, "😀é"), (4, 7, "😀é")]
    assert targets[0].id != targets[1].id
    assert targets == generate_targets(text, spec)


def test_literal_search_handles_overlaps_and_casefold_lengths():
    targets = generate_targets(
        "BANANA ß", TargetSpec(mode="literal", terms=["ana", "SS"], ignore_case=True),
    )
    assert [(t.start, t.end, t.text) for t in targets] == [(1, 4, "ANA"), (3, 6, "ANA"), (7, 8, "ß")]


def test_blank_document_and_missing_literal_yield_no_targets():
    assert generate_targets("\n ", TargetSpec(mode="document")) == []
    assert generate_targets("one", TargetSpec(mode="literal", terms=["two"])) == []


def test_candidate_overflow_is_an_error():
    with pytest.raises(ValueError, match="candidate"):
        generate_targets("aaaa", TargetSpec(mode="literal", terms=["a"], max_candidates=2))


def test_sentence_and_paragraph_boundaries_preserve_crlf_and_punctuation():
    text = "  振動。次は「正常！」\r\n  At 3.14. OK!\r\n\r\n\t次段落。"
    sentences = generate_targets(text, TargetSpec(mode="sentence"))
    assert [t.text for t in sentences] == ["振動。", "次は「正常！」", "At 3.14.", "OK!", "次段落。"]
    paragraphs = generate_targets(text, TargetSpec(mode="paragraph"))
    assert [t.text for t in paragraphs] == ["振動。次は「正常！」\r\n  At 3.14. OK!", "次段落。"]
    assert all(text[t.start:t.end] == t.text for t in sentences + paragraphs)


def test_regex_zero_length_matches_do_not_become_targets():
    targets = generate_targets("aba", TargetSpec(mode="regex", pattern="(?=a)|a"))
    assert [(t.start, t.end) for t in targets] == [(0, 1), (2, 3)]


def test_regex_runtime_has_a_timeout():
    with pytest.raises(ValueError, match="timed out"):
        generate_targets("a" * 50_000 + "!", TargetSpec(mode="regex", pattern="^(a+)+$"))


def _exhaust_regex_recursion(*args, **kwargs):
    raise RecursionError


def test_regex_spec_rejects_compiler_recursion(monkeypatch):
    monkeypatch.setattr(regex, "compile", _exhaust_regex_recursion)
    with pytest.raises(ValidationError, match="nested"):
        TargetSpec(mode="regex", pattern="a")


def test_regex_runtime_reports_compiler_recursion(monkeypatch):
    spec = TargetSpec(mode="regex", pattern="a")
    monkeypatch.setattr(regex, "finditer", _exhaust_regex_recursion)
    with pytest.raises(ValueError, match="nested"):
        generate_targets("a", spec)


def test_context_includes_entire_crossing_target_and_exposes_offsets():
    text = "😀 A。B。 C"
    target = Target(id="t", start=3, end=5, text="。B")
    context = build_context(text, target, ContextSpec(mode="sentence"))
    assert (context["start"], context["end"], context["text"]) == (0, 6, "😀 A。B。")
    assert context["text"][context["target_start"]:context["target_end"]] == "。B"


def test_context_overflow_is_an_error():
    text = "a😀bc"
    target = Target(id="t", start=1, end=2, text="😀")
    with pytest.raises(ValueError, match="context"):
        build_context(text, target, ContextSpec(mode="document", max_chars=2))


def test_misaligned_target_is_rejected_before_model_call():
    recipe = make_recipe()
    target = Target(id="t", start=1, end=2, text="a")
    with pytest.raises(ValueError, match="target"):
        annotate_target("ab", target, recipe, None)


class FixtureProvider:
    def __init__(self, *answers, origin="model"):
        self.responses = iter(answers)
        self.states = []
        self.origin = origin

    def evaluate(self, state, features):
        self.states.append(deepcopy(state))
        return ProviderResult(next(self.responses), "fixture", {}, self.origin)


def boolean_answer(probability):
    return {
        "relevant": {
            "value": None,
            "probability": probability,
            "probabilities": {"yes": probability, "no": 1 - probability},
            "confidence": None,
        },
    }


@pytest.mark.parametrize("probability, label, status", [
    (0.2, False, "excluded"),
    (0.200001, None, "review"),
    (0.799999, None, "review"),
    (0.8, True, "accepted"),
])
def test_noul_thresholds_keep_false_distinct_from_uncertainty(probability, label, status):
    provider = FixtureProvider(boolean_answer(probability))
    annotation = annotate_target("pump", Target(id="t", start=0, end=4, text="pump"), make_recipe(), provider)
    assert annotation["features"]["relevant"] is label
    assert annotation["status"] == status
    assert annotation["reviewed"] is False


def test_secondary_context_is_separate_and_disagreement_prevents_exclusion():
    provider = FixtureProvider(boolean_answer(0.1), boolean_answer(0.9))
    text = "bad pump"
    target = Target(id="t", start=4, end=8, text="pump")
    annotation = annotate_target(text, target, make_recipe(secondary_context=ContextSpec(mode="document")), provider)
    assert [state["text"] for state in provider.states] == ["pump", "bad pump"]
    assert all("answers" not in state for state in provider.states)
    assert [p["stage"] for p in annotation["predictions"]] == ["primary", "secondary"]
    assert annotation["status"] == "review"
    assert any("disagreement" in reason for reason in annotation["reasons"])


def test_secondary_context_with_identical_visible_text_skips_duplicate_inference():
    provider = FixtureProvider(boolean_answer(0.91))
    target = Target(id="t", start=0, end=4, text="pump")
    annotation = annotate_target("pump", target, make_recipe(secondary_context=ContextSpec(mode="document")), provider)
    assert len(provider.states) == 1
    assert len(annotation["predictions"]) == 1


def test_nonfilter_boolean_false_stays_accepted():
    recipe = make_recipe()
    feature = recipe.features[0].model_copy(update={"filter": False})
    recipe = recipe.model_copy(update={"features": [feature]})
    provider = FixtureProvider(boolean_answer(0.0))
    target = Target(id="t", start=0, end=4, text="pump")
    annotation = annotate_target("pump", target, recipe, provider)
    assert annotation["features"]["relevant"] is False
    assert annotation["status"] == "accepted"


@pytest.mark.parametrize("verified, status", [(False, "excluded"), (None, "review"), (True, "review")])
def test_verifier_uses_its_own_label_and_unknown_prevents_exclusion(verified, status):
    provider = FixtureProvider(boolean_answer(0.1))
    verifier = FixtureProvider({
        "relevant": {"value": verified, "probability": None, "probabilities": {}, "confidence": None},
    }, origin="verifier")
    target = Target(id="t", start=0, end=4, text="pump")
    annotation = annotate_target("pump", target, make_recipe(verifier=True), provider, verifier)
    assert [p["stage"] for p in annotation["predictions"]] == ["primary", "verifier"]
    assert verifier.states == provider.states
    assert annotation["status"] == status
    assert annotation["features"]["relevant"] is False


def test_missing_model_feature_is_an_error():
    with pytest.raises(ProviderError):
        annotate_target("pump", Target(id="t", start=0, end=4, text="pump"), make_recipe(), FixtureProvider({}))


def test_multi_label_abstains_if_any_label_is_uncertain():
    base = make_recipe()
    tags = FeatureSpec(id="tags", name="Tags", kind="multi_label", instruction="Identify stated fault types.",
                       choices={"heat": "Heat", "power": "Power"})
    recipe = base.model_copy(update={"features": [tags]})
    provider = FixtureProvider({
        "tags": {"value": None, "probabilities": {"heat": 0.97, "power": 0.5}, "confidence": None},
    })
    annotation = annotate_target("pump", Target(id="t", start=0, end=4, text="pump"), recipe, provider)
    assert annotation["features"]["tags"] is None
    assert annotation["status"] == "review"


def test_score_minimum_is_a_review_rule():
    score = FeatureSpec(id="level", name="Level", kind="score", instruction="Score the inspection priority.",
                        levels=["routine", "immediate"], minimum=0.8)
    recipe = make_recipe().model_copy(update={"features": [score]})
    provider = FixtureProvider({
        "level": {"value": 0.25, "probabilities": {"0": 0.75, "1": 0.25}, "confidence": 0.95},
    })
    annotation = annotate_target("pump", Target(id="t", start=0, end=4, text="pump"), recipe, provider)
    assert annotation["features"]["level"] == 0.25
    assert annotation["status"] == "review"
    assert "below_minimum:level" in annotation["reasons"]


def test_deterministic_audit_can_select_both_filter_states():
    recipe = make_recipe(audit_fraction=0.4)

    def selected():
        results = []
        for index in range(40):
            text = f"pump {index}"
            target = Target(id="t", start=0, end=len(text), text=text)
            provider = FixtureProvider(boolean_answer(0.01 if index % 2 else 0.99))
            annotation = annotate_target(text, target, recipe, provider)
            if "audit" in annotation["reasons"]:
                assert annotation["status"] == "review"
                results.append(index)
        return results

    first = selected()
    assert 0 < len(first) < 40
    assert {index % 2 for index in first} == {0, 1}
    assert first == selected()


def test_demo_always_requires_review_even_if_a_filter_would_exclude():
    recipe = make_recipe()
    annotation = annotate_target("pump", Target(id="t", start=0, end=4, text="pump"), recipe, DemoProvider())
    assert annotation["status"] == "review"
    assert annotation["origin"] == "demo"
    assert annotation["predictions"][0]["model"] == "demo-v1"
