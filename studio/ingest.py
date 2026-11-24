"""Bounded input readers. Keep raw input unchanged and preserve values."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path


def _reject_constant(_value):
    raise ValueError("Non-finite JSON numbers are not supported.")


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON object keys are not supported.")
        obj[key] = value
    return obj


def _jsonable(value):
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Unpaired Unicode surrogates are not supported.") from error
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite numeric values are not supported.")
        return value
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("Column names must be strings.")
        for key in value:
            _jsonable(key)
        return {k: _jsonable(v) for k, v in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise ValueError(f"Unsupported input value type: {type(value).__name__}")


def parse_upload(filename: str, data: bytes, settings) -> tuple[list[dict], list[str], str]:
    if not data:
        raise ValueError("ファイルが空です / The file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise ValueError("アップロード上限を超えています / Upload too large.")
    suffix = Path(filename).suffix.lower()
    rows = []
    if suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        source = pq.ParquetFile(pa.BufferReader(data))
        if source.metadata.num_rows > settings.max_rows:
            raise ValueError("Too many rows.")
        # Compressed size alone does not bound decoded memory.
        decoded_size = sum(source.metadata.row_group(i).total_byte_size for i in range(source.num_row_groups))
        if decoded_size > settings.max_upload_bytes * 5:
            raise ValueError("Decoded Parquet data exceeds the memory limit.")
        rows = [_jsonable(r) for r in source.read().to_pylist()]
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("UTF-8で保存して再投入してください / Use UTF-8 encoding.") from error
        if suffix == ".csv":
            reader = csv.reader(io.StringIO(text, newline=""), strict=True)
            header = next(reader, [])
            if not header or any(not h for h in header) or len(set(header)) != len(header):
                raise ValueError("CSV needs nonempty, unique column names.")
            for line, values in enumerate(reader, 2):
                if not values:
                    continue
                if len(values) != len(header):
                    raise ValueError(f"CSV row {line} has a different number of columns.")
                rows.append(dict(zip(header, values, strict=True)))
                if len(rows) > settings.max_rows:
                    raise ValueError("Too many rows.")
        elif suffix in {".jsonl", ".ndjson"}:
            for number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object))
                except (ValueError, RecursionError) as error:
                    raise ValueError(f"Invalid JSON on line {number}.") from error
                if len(rows) > settings.max_rows:
                    raise ValueError("Too many rows.")
        elif suffix == ".json":
            try:
                parsed = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
            except RecursionError as error:
                raise ValueError("JSON nesting exceeds the supported depth.") from error
            rows = parsed if isinstance(parsed, list) else [parsed]
        elif suffix in {".txt", ".md"}:
            rows = [{"id": "document-1", "text": text}]
        else:
            raise ValueError("Supported files: CSV, JSON, JSONL, TXT, Markdown, Parquet.")
    if not rows or len(rows) > settings.max_rows:
        raise ValueError("The file must contain between 1 and the configured maximum number of rows.")
    columns = []
    for row in rows:
        if not isinstance(row, dict) or not row:
            raise ValueError("Each row must be a nonempty JSON object.")
        try:
            _jsonable(row)
        except RecursionError as error:
            raise ValueError("JSON nesting exceeds the supported depth.") from error
        for key in row:
            if not isinstance(key, str) or not key or len(key) > 200:
                raise ValueError("Column names must be nonempty strings of at most 200 characters.")
            if key not in columns:
                columns.append(key)
        if len(columns) > 128:
            raise ValueError("At most 128 columns are supported.")
    return rows, columns, hashlib.sha256(data).hexdigest()


def validate_source(records, recipe, settings):
    seen = set()
    for record in records:
        values = record.values
        text = values.get(recipe.source.text_field)
        if not isinstance(text, str):
            raise ValueError(f"Record {record.position + 1}: text field must contain a string.")
        if len(text) > settings.max_text_chars:
            raise ValueError(f"Record {record.position + 1}: text exceeds the configured limit.")
        if recipe.source.id_field:
            raw = values.get(recipe.source.id_field)
            if raw is None or isinstance(raw, (list, dict, bool)) or str(raw) == "":
                raise ValueError(f"Record {record.position + 1}: ID must be a nonempty string or number.")
            if str(raw) in seen:
                raise ValueError("ID values are not unique. Choose another column or automatic IDs.")
            seen.add(str(raw))
        if recipe.source.group_field:
            group = values.get(recipe.source.group_field)
            if group is None or isinstance(group, (list, dict, bool)) or str(group) == "":
                raise ValueError("Group values must be nonempty strings or numbers.")


def source_id(record, recipe) -> str:
    return str(record.values[recipe.source.id_field]) if recipe.source.id_field else f"row-{record.position + 1}"
