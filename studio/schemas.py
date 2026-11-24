"""Typed contracts shared by target generation, inference, and review."""

import hashlib
import json
from typing import Annotated, Literal, Self

import regex
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

TargetMode = Literal["document", "sentence", "paragraph", "literal", "regex"]
ContextMode = Literal["target", "sentence", "window", "paragraph", "document"]
FeatureKind = Literal["choice", "multi_label", "boolean", "score"]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("a nonblank string is required")
    return value


DisplayText = Annotated[str, StringConstraints(min_length=1, max_length=256), AfterValidator(_nonblank)]
Description = Annotated[str, StringConstraints(min_length=1, max_length=4_000), AfterValidator(_nonblank)]
FeatureID = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True, frozen=True)


class SourceSpec(_StrictModel):
    text_field: DisplayText = "text"
    id_field: DisplayText | None = None
    group_field: DisplayText | None = None


class TargetSpec(_StrictModel):
    mode: TargetMode = "sentence"
    terms: Annotated[list[DisplayText], Field(max_length=64)] = Field(default_factory=list)
    pattern: Annotated[str, StringConstraints(min_length=1, max_length=2_000)] | None = None
    ignore_case: bool = False
    max_candidates: Annotated[int, Field(ge=1, le=2_000)] = 200

    @model_validator(mode="after")
    def check_target_mode(self) -> Self:
        if self.mode == "literal" and not self.terms:
            raise ValueError("literal targets require at least one term")
        if self.mode == "regex":
            if self.pattern is None:
                raise ValueError("regex targets require a pattern")
            try:
                regex.compile(self.pattern, regex.VERSION1)
            except regex.error:
                raise ValueError("invalid target regex pattern") from None
            except RecursionError:
                raise ValueError("target regex pattern is too deeply nested") from None
        return self


class ContextSpec(_StrictModel):
    mode: ContextMode = "window"
    window_chars: Annotated[int, Field(ge=0, le=120_000)] = 180
    max_chars: Annotated[int, Field(ge=1, le=120_000)] = 12_000


class FeatureSpec(_StrictModel):
    id: FeatureID
    name: DisplayText
    kind: FeatureKind
    instruction: Description
    choices: Annotated[dict[DisplayText, Description], Field(max_length=32)] = Field(default_factory=dict)
    levels: Annotated[list[Description], Field(max_length=10)] = Field(default_factory=list)
    filter: bool = False
    minimum: float | None = None

    @model_validator(mode="after")
    def check_feature_kind(self) -> Self:
        instruction = self.instruction.strip().casefold()
        if instruction in {self.id.casefold(), self.name.strip().casefold()}:
            raise ValueError("instruction must describe what to judge beyond the feature name")
        if self.kind in {"choice", "multi_label"}:
            required_choices = 2 if self.kind == "choice" else 1
            if len(self.choices) < required_choices:
                raise ValueError(f"{self.kind} requires at least {required_choices} choice descriptions")
        elif self.choices:
            raise ValueError("choices are only allowed for choice and multi_label features")
        if self.kind == "score":
            if len(self.levels) < 2 or len(set(self.levels)) != len(self.levels):
                raise ValueError("score requires 2 to 10 distinct level descriptions")
            if self.minimum is not None and not 0 <= self.minimum <= len(self.levels) - 1:
                raise ValueError("minimum must be within the score levels")
        elif self.levels or self.minimum is not None:
            raise ValueError("levels and minimum are only allowed for score features")
        if self.filter and self.kind != "boolean":
            raise ValueError("filter is only allowed for boolean features")
        return self

    def validate_value(self, value: object) -> object:
        """Validate a prediction or review value, preserving an explicit unknown."""
        if value is None:
            return value
        if self.kind == "choice":
            valid = isinstance(value, str) and value in self.choices
        elif self.kind == "multi_label":
            valid = (
                isinstance(value, list)
                and all(isinstance(label, str) and label in self.choices for label in value)
                and len(set(value)) == len(value)
            )
        elif self.kind == "boolean":
            valid = isinstance(value, bool)
        else:
            valid = type(value) in {int, float} and 0 <= value <= len(self.levels) - 1
        if not valid:
            raise ValueError(f"invalid {self.kind} value for feature {self.id}")
        return value


class QualitySpec(_StrictModel):
    min_confidence: Probability = 0.6
    boolean_low: Annotated[float, Field(ge=0, lt=0.5)] = 0.2
    boolean_high: Annotated[float, Field(gt=0.5, le=1)] = 0.8
    secondary_context: ContextSpec | None = None
    verifier: bool = False
    score_tolerance: Annotated[float, Field(ge=0.0, le=9.0)] = 0.5
    audit_fraction: Probability = 0.05


class Recipe(_StrictModel):
    version: Annotated[int, Field(ge=1, le=1)] = 1
    name: DisplayText = "Annotation recipe"
    provider: Literal["demo", "jev"] = "demo"
    model: DisplayText = "jev-1.14.0"
    source: SourceSpec = Field(default_factory=SourceSpec)
    target: TargetSpec = Field(default_factory=TargetSpec)
    context: ContextSpec = Field(default_factory=ContextSpec)
    features: Annotated[list[FeatureSpec], Field(min_length=1, max_length=32)]
    quality: QualitySpec = Field(default_factory=QualitySpec)

    @model_validator(mode="after")
    def check_feature_ids(self) -> Self:
        if len({feature.id for feature in self.features}) != len(self.features):
            raise ValueError("feature IDs must be unique")
        return self


class Target(_StrictModel):
    id: DisplayText
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]
    text: Annotated[str, StringConstraints(min_length=1)]

    @model_validator(mode="after")
    def check_span_length(self) -> Self:
        if self.end - self.start != len(self.text):
            raise ValueError("target text must have the same length as its span")
        return self


def _canonical_model(value: object) -> dict:
    if isinstance(value, FeatureSpec):
        data = value.model_dump(mode="json")
        # A criteria map carries order in this feature schema.
        data["choices"] = list(value.choices.items())
        return data
    if isinstance(value, Recipe):
        data = value.model_dump(mode="json")
        data["features"] = [_canonical_model(feature) for feature in value.features]
        return data
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"cannot hash objects of type {type(value).__name__}")


def canonical_hash(value: object) -> str:
    """Hash JSON content deterministically, including the order of recipe choices."""
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"), default=_canonical_model,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
