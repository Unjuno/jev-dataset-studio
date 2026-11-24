"""Generate and expand half-open Unicode codepoint spans."""

from collections.abc import Iterator

import regex

from .schemas import ContextSpec, Target, TargetSpec, canonical_hash

_REGEX_TIMEOUT_SECONDS = 0.1
_SENTENCE_BREAK = regex.compile(
    r"""(?:[!?。！？．]+|\.{2,}|\.(?=$|[\s"'”’»」』】）〕〉》)\]!?]|[\p{Han}\p{Hiragana}\p{Katakana}]))"""
    r"""[.!?。！？．]*["'”’»」』】）〕〉》)\]]*|\R+""",
    regex.VERSION1,
)
_PARAGRAPH_BREAK = regex.compile(r"(?:\R[^\S\n\r\v\f\x85\u2028\u2029]*){2,}|\u2029", regex.VERSION1)


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _segments(text: str, mode: str) -> Iterator[tuple[int, int]]:
    breaks = _SENTENCE_BREAK if mode == "sentence" else _PARAGRAPH_BREAK
    start = 0
    for match in breaks.finditer(text):
        separator = match.group()[0].isspace()
        end = match.start() if separator else match.end()
        left, right = _trimmed_span(text, start, end)
        if left < right:
            yield left, right
        start = match.end()
    left, right = _trimmed_span(text, start, len(text))
    if left < right:
        yield left, right


def _match_spans(text: str, spec: TargetSpec) -> Iterator[tuple[int, int]]:
    flags = regex.VERSION1 | (regex.IGNORECASE if spec.ignore_case else 0)
    if spec.mode == "literal":
        for term in spec.terms:
            pattern = regex.compile(regex.escape(term), flags)
            for match in pattern.finditer(text, overlapped=True):
                yield match.span()
        return
    try:
        for match in regex.finditer(spec.pattern, text, flags, timeout=_REGEX_TIMEOUT_SECONDS):
            if match.start() < match.end():
                yield match.span()
    except TimeoutError:
        raise ValueError("target regex evaluation timed out") from None
    except regex.error:
        raise ValueError("invalid target regex pattern") from None
    except RecursionError:
        raise ValueError("target regex pattern is too deeply nested") from None


def generate_targets(text: str, spec: TargetSpec) -> list[Target]:
    """Find every matching span, or report candidate overflow."""
    if spec.mode == "document":
        spans = [(0, len(text))] if text.strip() else []
    elif spec.mode in {"sentence", "paragraph"}:
        spans = _segments(text, spec.mode)
    else:
        spans = _match_spans(text, spec)
    unique_spans = set()
    for span in spans:
        unique_spans.add(span)
        if len(unique_spans) > spec.max_candidates:
            raise ValueError(f"target candidate count exceeds the configured limit of {spec.max_candidates}")
    if not unique_spans:
        return []
    text_hash = canonical_hash(text)
    return [
        Target(id=f"{text_hash}:{start}:{end}", start=start, end=end, text=text[start:end])
        for start, end in sorted(unique_spans)
    ]


def build_context(text: str, target: Target, spec: ContextSpec) -> dict:
    """Return the chosen original context and both absolute and relative offsets."""
    if target.end > len(text) or text[target.start:target.end] != target.text:
        raise ValueError("target span does not match the source text")
    start, end = target.start, target.end
    if spec.mode == "document":
        start, end = 0, len(text)
    elif spec.mode == "window":
        start = max(0, start - spec.window_chars)
        end = min(len(text), end + spec.window_chars)
    elif spec.mode in {"sentence", "paragraph"}:
        for left, right in _segments(text, spec.mode):
            if left >= target.end:
                break
            if right > target.start:
                start = min(start, left)
                end = max(end, right)
    if end - start > spec.max_chars:
        raise ValueError(f"context length {end - start} exceeds the configured limit of {spec.max_chars}")
    return {
        "mode": spec.mode,
        "text": text[start:end],
        "start": start,
        "end": end,
        "target": target.model_dump(),
        "target_start": target.start - start,
        "target_end": target.end - start,
    }
