from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(min_length=1, max_length=10_000)
    course: str | None = Field(default=None, max_length=50)
    feedback_date: date | None = Field(default=None, alias="date")


class TaskResult(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    confidence_is_calibrated: bool


class AnalyzeResponse(BaseModel):
    id: str
    analysis_id: str
    text: str
    sentiment: TaskResult
    topic: TaskResult
    model_backend: str
    course: str | None
    feedback_date: date | None
    created_at: datetime


class FeedbackItem(BaseModel):
    id: str
    analysis_id: str
    text: str
    sentiment: str
    sentiment_confidence: float
    sentiment_probabilities: dict[str, float]
    topic: str
    topic_confidence: float
    topic_probabilities: dict[str, float]
    model_backend: str
    course: str | None
    feedback_date: date | None
    batch_id: str | None
    created_at: datetime


class FeedbackPage(BaseModel):
    items: list[FeedbackItem]
    page: int
    page_size: int
    total: int


class BatchResponse(BaseModel):
    batch_id: str
    filename: str
    total_records: int
    processed_records: int
    failed_records: int
    status: str
    results: list[FeedbackItem]


class CountItem(BaseModel):
    label: str
    count: int


class TopicSentimentItem(BaseModel):
    topic: str
    sentiments: dict[str, int]


class TrendItem(BaseModel):
    date: date
    sentiments: dict[str, int]


class SummaryResponse(BaseModel):
    total_feedback: int
    sentiments: list[CountItem]
    topics: list[CountItem]
    courses: list[CountItem]
    sentiment_by_topic: list[TopicSentimentItem]
    daily_trend: list[TrendItem]
    last_analyzed_at: datetime | None


class HealthResponse(BaseModel):
    status: str
    model_backend: str
    database: str
