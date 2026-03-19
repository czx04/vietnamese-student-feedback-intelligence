from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date

from ml.data import clean_text


@dataclass(frozen=True)
class BatchRow:
    text: str
    course: str | None = None
    feedback_date: date | None = None


def parse_feedback_csv(content: bytes, max_bytes: int, max_rows: int) -> list[BatchRow]:
    if len(content) > max_bytes:
        raise ValueError(f"CSV is larger than the {max_bytes // (1024 * 1024)} MB limit")
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must use UTF-8 encoding") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames or "text" not in reader.fieldnames:
        raise ValueError("CSV must contain a 'text' column")
    unsupported = set(reader.fieldnames).difference({"text", "course", "date"})
    if unsupported:
        raise ValueError(f"Unsupported CSV columns: {', '.join(sorted(unsupported))}")

    rows: list[BatchRow] = []
    for line_number, raw in enumerate(reader, start=2):
        if len(rows) >= max_rows:
            raise ValueError(f"CSV exceeds the {max_rows}-row limit")
        text = clean_text(raw.get("text", ""))
        if not text:
            raise ValueError(f"Row {line_number}: text must not be empty")
        if len(text) > 10_000:
            raise ValueError(f"Row {line_number}: text exceeds 10,000 characters")
        raw_date = clean_text(raw.get("date", ""))
        try:
            parsed_date = date.fromisoformat(raw_date) if raw_date else None
        except ValueError as exc:
            raise ValueError(f"Row {line_number}: date must use YYYY-MM-DD") from exc
        course = clean_text(raw.get("course", "")) or None
        if course and len(course) > 50:
            raise ValueError(f"Row {line_number}: course exceeds 50 characters")
        rows.append(BatchRow(text=text, course=course, feedback_date=parsed_date))

    if not rows:
        raise ValueError("CSV contains no feedback rows")
    return rows
