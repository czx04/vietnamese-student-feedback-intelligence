from pathlib import Path
import asyncio

import httpx

from backend.app.config import Settings
from backend.app.main import create_app
from ml.inference.types import FeedbackPrediction, TaskPrediction


class FakePredictor:
    backend_name = "fake"

    def predict(self, text: str) -> FeedbackPrediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts) -> list[FeedbackPrediction]:
        return [
            FeedbackPrediction(
                text=text.strip(),
                sentiment=TaskPrediction(
                    label="positive",
                    confidence=0.8,
                    probabilities={"negative": 0.1, "neutral": 0.1, "positive": 0.8},
                ),
                topic=TaskPrediction(
                    label="lecturer",
                    confidence=0.7,
                    probabilities={
                        "lecturer": 0.7,
                        "program": 0.1,
                        "facility": 0.1,
                        "others": 0.1,
                    },
                ),
                model_backend="fake",
            )
            for text in texts
        ]


def build_app(tmp_path: Path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}")
    return create_app(settings=settings, predictor_factory=FakePredictor)


def request(app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_analyze_persists_and_updates_analytics(tmp_path: Path):
    app = build_app(tmp_path)

    response = request(
        app,
        "POST",
        "/api/v1/analyze",
        json={"text": "Giảng viên dạy tốt", "course": "AI", "date": "2026-10-01"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["sentiment"]["label"] == "positive"
    assert result["topic"]["label"] == "lecturer"
    assert request(app, "GET", f"/api/v1/feedback/{result['id']}").status_code == 200
    assert request(app, "GET", "/api/v1/feedback").json()["total"] == 1
    summary = request(app, "GET", "/api/v1/analytics/summary").json()
    assert summary["total_feedback"] == 1
    assert summary["sentiments"] == [{"label": "positive", "count": 1}]
    assert summary["sentiment_by_topic"] == [
        {"topic": "lecturer", "sentiments": {"positive": 1}}
    ]
    assert summary["daily_trend"] == [
        {"date": "2026-10-01", "sentiments": {"positive": 1}}
    ]
    assert request(app, "GET", "/api/v1/feedback?min_confidence=0.9").json()[
        "total"
    ] == 0


def test_batch_csv_validates_and_persists(tmp_path: Path):
    app = build_app(tmp_path)
    csv_content = (
        "text,course,date\n"
        "Giảng viên nhiệt tình,AI,2026-10-01\n"
        "Phòng học sạch sẽ,Database,2026-10-02\n"
    )

    response = request(
        app,
        "POST",
        "/api/v1/analyze/batch",
        files={"file": ("feedback.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["processed_records"] == 2
    assert request(app, "GET", "/api/v1/feedback").json()["total"] == 2


def test_batch_csv_rejects_invalid_schema(tmp_path: Path):
    app = build_app(tmp_path)
    response = request(
        app,
        "POST",
        "/api/v1/analyze/batch",
        files={"file": ("bad.csv", "comment\nhello\n", "text/csv")},
    )
    assert response.status_code == 422
    assert "'text' column" in response.json()["detail"]
