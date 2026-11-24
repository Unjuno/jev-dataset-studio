"""Evaluate a target and preserve each prediction that informs its review status."""

from copy import deepcopy

from .providers import ProviderError, validate_provider_result
from .schemas import FeatureSpec, QualitySpec, Recipe, Target, canonical_hash
from .targets import build_context


def _decode_boolean(probability: float, quality: QualitySpec) -> bool | None:
    if probability <= quality.boolean_low:
        return False
    if probability >= quality.boolean_high:
        return True
    return None


def _decode(answer: dict, feature: FeatureSpec, quality: QualitySpec, origin: str):
    if origin == "verifier":
        return answer["value"]
    if feature.kind == "boolean":
        return _decode_boolean(answer["probability"], quality)
    if feature.kind == "multi_label":
        labels = {label: _decode_boolean(answer["probabilities"][label], quality) for label in feature.choices}
        return None if None in labels.values() else [label for label, present in labels.items() if present]
    return answer["value"]


def _disagrees(primary, other, feature: FeatureSpec, tolerance: float) -> bool:
    if primary is None or other is None:
        return primary is not other
    if feature.kind == "score":
        return abs(primary - other) > tolerance
    if feature.kind == "multi_label":
        return set(primary) != set(other)
    return primary != other


def _audit_selected(text: str, target: Target, recipe: Recipe) -> bool:
    fraction = recipe.quality.audit_fraction
    if not fraction:
        return False
    sample = canonical_hash({"text": text, "target": target, "recipe": recipe})[:16]
    return int(sample, 16) < fraction * 2**64


def _quality_reasons(stage: str, value, answer: dict, origin: str, feature: FeatureSpec, quality: QualitySpec):
    detail = feature.id if stage == "primary" else f"{stage}:{feature.id}"
    if value is None:
        yield f"uncertain:{detail}"
    if origin != "verifier" and feature.kind in {"choice", "score"} and answer["confidence"] < quality.min_confidence:
        yield f"low_confidence:{detail}"
    if feature.minimum is not None and value is not None and value < feature.minimum:
        yield f"below_minimum:{detail}"


def annotate_target(text: str, target: Target, recipe: Recipe, provider, verifier=None) -> dict:
    """Return an annotation with its accepted, review, or excluded decision."""
    if recipe.quality.verifier and verifier is None:
        raise ProviderError("recipe requires a configured verifier")
    primary = build_context(text, target, recipe.context)
    stages = [("primary", primary, provider)]
    if recipe.quality.secondary_context is not None:
        secondary = build_context(text, target, recipe.quality.secondary_context)
        if (primary["start"], primary["end"]) != (secondary["start"], secondary["end"]):
            stages.append(("secondary", secondary, provider))
    if recipe.quality.verifier:
        stages.append(("verifier", primary, verifier))

    predictions = []
    feature_values = {}
    reasons = []
    for stage, context, evaluator in stages:
        result = evaluator.evaluate(deepcopy(context), recipe.features)
        validate_provider_result(result, recipe.features)
        predictions.append({
            "stage": stage,
            "context": context,
            "model": result.model,
            "origin": result.origin,
            "answers": deepcopy(result.answers),
            "usage": deepcopy(result.usage),
        })
        for feature in recipe.features:
            answer = result.answers[feature.id]
            value = _decode(answer, feature, recipe.quality, result.origin)
            reasons.extend(_quality_reasons(stage, value, answer, result.origin, feature, recipe.quality))
            if stage == "primary":
                feature_values[feature.id] = value
            elif _disagrees(feature_values[feature.id], value, feature, recipe.quality.score_tolerance):
                reasons.append(f"disagreement:{stage}:{feature.id}")

    origin = predictions[0]["origin"]
    if origin == "demo":
        reasons.append("demo")
    if _audit_selected(text, target, recipe):
        reasons.append("audit")
    negatives = [f"filter_negative:{feature.id}" for feature in recipe.features
                 if feature.filter and feature_values[feature.id] is False]
    status = "review" if reasons else "excluded" if negatives else "accepted"
    return {
        "target": target.model_dump(),
        "features": feature_values,
        "predictions": predictions,
        "status": status,
        "reasons": reasons + negatives,
        "origin": origin,
        "reviewed": False,
    }
