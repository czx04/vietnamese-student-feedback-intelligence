import pytest

from backend.app.services import parse_feedback_csv


def test_parse_feedback_csv_supports_optional_columns():
    rows = parse_feedback_csv(
        "text,course,date\nBài giảng tốt,AI,2026-10-09\n".encode(),
        max_bytes=1024,
        max_rows=10,
    )
    assert rows[0].text == "Bài giảng tốt"
    assert rows[0].course == "AI"
    assert rows[0].feedback_date.isoformat() == "2026-10-09"


@pytest.mark.parametrize(
    "content,message",
    [
        (b"comment\nhello\n", "'text' column"),
        (b"text,date\nhello,09/10/2026\n", "YYYY-MM-DD"),
        (b"text\n\n", "no feedback rows"),
    ],
)
def test_parse_feedback_csv_rejects_bad_input(content, message):
    with pytest.raises(ValueError, match=message):
        parse_feedback_csv(content, max_bytes=1024, max_rows=10)
