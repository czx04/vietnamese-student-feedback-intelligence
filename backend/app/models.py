from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


def new_uuid() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisBatch(Base):
    __tablename__ = "analysis_batches"
    __table_args__ = (
        CheckConstraint(
            "source_type IN ('manual', 'csv', 'api')", name="ck_batch_source_type"
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'partial', 'failed')",
            name="ck_batch_status",
        ),
        CheckConstraint(
            "total_records >= 0 AND processed_records >= 0 AND failed_records >= 0",
            name="ck_batch_nonnegative_counts",
        ),
        CheckConstraint(
            "processed_records + failed_records <= total_records",
            name="ck_batch_count_limit",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    filename: Mapped[str | None] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(20), default="csv")
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    total_records: Mapped[int] = mapped_column(Integer, default=0)
    processed_records: Mapped[int] = mapped_column(Integer, default=0)
    failed_records: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="batch")


class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("batch_id", "source_row_number", name="uq_feedback_batch_row"),
        CheckConstraint(
            "source_row_number IS NULL OR source_row_number > 0",
            name="ck_feedback_positive_row",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    batch_id: Mapped[str | None] = mapped_column(
        ForeignKey("analysis_batches.id", ondelete="SET NULL"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    feedback_date: Mapped[date | None] = mapped_column(Date)
    course_code: Mapped[str | None] = mapped_column(String(50), index=True)
    source_row_number: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    batch: Mapped[AnalysisBatch | None] = relationship(back_populates="feedback")
    analyses: Mapped[list["FeedbackAnalysis"]] = relationship(
        back_populates="feedback", cascade="all, delete-orphan"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("id", "task", name="uq_model_id_task"),
        UniqueConstraint("task", "name", "version", name="uq_model_version"),
        CheckConstraint("task IN ('sentiment', 'topic')", name="ck_model_task"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    task: Mapped[str] = mapped_column(String(20), index=True)
    name: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(50))
    model_type: Mapped[str] = mapped_column(String(50), index=True)
    artifact_uri: Mapped[str] = mapped_column(Text)
    labels: Mapped[list] = mapped_column(JSON)
    validation_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    test_metrics: Mapped[dict | None] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class FeedbackAnalysis(Base):
    __tablename__ = "feedback_analyses"
    __table_args__ = (
        ForeignKeyConstraint(
            ["sentiment_model_id", "sentiment_model_task"],
            ["model_versions.id", "model_versions.task"],
            name="fk_analysis_sentiment_model",
        ),
        ForeignKeyConstraint(
            ["topic_model_id", "topic_model_task"],
            ["model_versions.id", "model_versions.task"],
            name="fk_analysis_topic_model",
        ),
        CheckConstraint(
            "sentiment_model_task = 'sentiment'", name="ck_sentiment_model_task"
        ),
        CheckConstraint("topic_model_task = 'topic'", name="ck_topic_model_task"),
        CheckConstraint(
            "status IN ('processing', 'completed', 'failed')",
            name="ck_analysis_status",
        ),
        CheckConstraint(
            "sentiment_confidence IS NULL OR "
            "(sentiment_confidence >= 0 AND sentiment_confidence <= 1)",
            name="ck_sentiment_confidence",
        ),
        CheckConstraint(
            "topic_confidence IS NULL OR (topic_confidence >= 0 AND topic_confidence <= 1)",
            name="ck_topic_confidence",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    feedback_id: Mapped[str] = mapped_column(
        ForeignKey("feedback.id", ondelete="CASCADE"), index=True
    )
    sentiment_model_id: Mapped[str] = mapped_column(String(36))
    sentiment_model_task: Mapped[str] = mapped_column(String(20), default="sentiment")
    topic_model_id: Mapped[str] = mapped_column(String(36))
    topic_model_task: Mapped[str] = mapped_column(String(20), default="topic")
    sentiment_label: Mapped[str | None] = mapped_column(String(20), index=True)
    sentiment_confidence: Mapped[float | None] = mapped_column(Numeric(6, 5))
    sentiment_score_type: Mapped[str] = mapped_column(String(30))
    sentiment_scores: Mapped[dict | None] = mapped_column(JSON)
    topic_label: Mapped[str | None] = mapped_column(String(20), index=True)
    topic_confidence: Mapped[float | None] = mapped_column(Numeric(6, 5))
    topic_score_type: Mapped[str] = mapped_column(String(30))
    topic_scores: Mapped[dict | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="processing", index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    feedback: Mapped[Feedback] = relationship(back_populates="analyses")
