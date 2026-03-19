from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from typing import Callable

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from backend.app.config import Settings
from backend.app.database import build_engine, build_session_factory, session_dependency
from backend.app.models import AnalysisBatch, Feedback, FeedbackAnalysis, ModelVersion
from backend.app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchResponse,
    CountItem,
    FeedbackItem,
    FeedbackPage,
    HealthResponse,
    SummaryResponse,
    TaskResult,
    TopicSentimentItem,
    TrendItem,
)
from backend.app.services import parse_feedback_csv
from backend.migrations.migration_001_initial_schema import upgrade
from ml.inference import FeedbackPredictor
from ml.phobert_common import TARGETS


def _score_type(task_prediction) -> str:
    return (
        "calibrated_probability"
        if task_prediction.confidence_is_calibrated
        else "uncalibrated_probability"
    )


def _to_analysis(
    prediction,
    feedback_id: str,
    model_ids: dict[str, str],
    latency_ms: int,
) -> FeedbackAnalysis:
    return FeedbackAnalysis(
        feedback_id=feedback_id,
        sentiment_model_id=model_ids["sentiment"],
        sentiment_model_task="sentiment",
        topic_model_id=model_ids["topic"],
        topic_model_task="topic",
        sentiment_label=prediction.sentiment.label,
        sentiment_confidence=prediction.sentiment.confidence,
        sentiment_score_type=_score_type(prediction.sentiment),
        sentiment_scores=prediction.sentiment.probabilities,
        topic_label=prediction.topic.label,
        topic_confidence=prediction.topic.confidence,
        topic_score_type=_score_type(prediction.topic),
        topic_scores=prediction.topic.probabilities,
        status="completed",
        latency_ms=latency_ms,
        completed_at=datetime.now(timezone.utc),
    )


def _task_result(label, confidence, scores, score_type) -> TaskResult:
    return TaskResult(
        label=label,
        confidence=confidence,
        probabilities=scores,
        confidence_is_calibrated=score_type == "calibrated_probability",
    )


def _analyze_response(
    feedback: Feedback,
    analysis: FeedbackAnalysis,
    model_backend: str,
) -> AnalyzeResponse:
    return AnalyzeResponse(
        id=feedback.id,
        analysis_id=analysis.id,
        text=feedback.text,
        sentiment=_task_result(
            analysis.sentiment_label,
            analysis.sentiment_confidence,
            analysis.sentiment_scores,
            analysis.sentiment_score_type,
        ),
        topic=_task_result(
            analysis.topic_label,
            analysis.topic_confidence,
            analysis.topic_scores,
            analysis.topic_score_type,
        ),
        model_backend=model_backend,
        course=feedback.course_code,
        feedback_date=feedback.feedback_date,
        created_at=feedback.created_at,
    )


def _feedback_item(
    feedback: Feedback,
    analysis: FeedbackAnalysis,
    model_backend: str,
) -> FeedbackItem:
    return FeedbackItem(
        id=feedback.id,
        analysis_id=analysis.id,
        text=feedback.text,
        sentiment=analysis.sentiment_label,
        sentiment_confidence=analysis.sentiment_confidence,
        sentiment_probabilities=analysis.sentiment_scores,
        topic=analysis.topic_label,
        topic_confidence=analysis.topic_confidence,
        topic_probabilities=analysis.topic_scores,
        model_backend=model_backend,
        course=feedback.course_code,
        feedback_date=feedback.feedback_date,
        batch_id=feedback.batch_id,
        created_at=feedback.created_at,
    )


def _seed_active_models(
    session: Session,
    backend_name: str,
    artifacts_dir,
) -> dict[str, str]:
    display_name = "PhoBERT Base" if backend_name == "phobert" else "Linear SVM"
    version = "step9" if backend_name == "phobert" else "baseline-v1"
    model_ids: dict[str, str] = {}
    report_root = artifacts_dir.parent / "reports"
    for task, labels in TARGETS.items():
        session.execute(
            update(ModelVersion)
            .where(ModelVersion.task == task, ModelVersion.is_active.is_(True))
            .values(is_active=False)
        )
        model = session.scalar(
            select(ModelVersion).where(
                ModelVersion.task == task,
                ModelVersion.name == display_name,
                ModelVersion.version == version,
            )
        )
        artifact = (
            artifacts_dir / "phobert" / task / "best_model"
            if backend_name == "phobert"
            else artifacts_dir / f"{task}_model.joblib"
        )
        if backend_name == "phobert":
            validation_path = report_root / "phobert" / task / "validation_metrics.json"
            test_path = report_root / "phobert" / task / "test_metrics.json"
        else:
            validation_path = report_root / "baseline" / f"linear_svm_{task}_validation_metrics.json"
            test_path = report_root / "comparison" / f"linear_svm_{task}_test_metrics.json"

        def read_metrics(path):
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        if model is None:
            model = ModelVersion(
                task=task,
                name=display_name,
                version=version,
                model_type=backend_name,
                artifact_uri=str(artifact),
                labels=labels,
                validation_metrics=read_metrics(validation_path),
                test_metrics=read_metrics(test_path),
                is_active=True,
            )
            session.add(model)
            session.flush()
        else:
            model.is_active = True
            model.artifact_uri = str(artifact)
            model.labels = labels
            model.validation_metrics = read_metrics(validation_path)
            model.test_metrics = read_metrics(test_path)
        model_ids[task] = model.id
    return model_ids


def _latest_completed_analysis_id():
    return (
        select(FeedbackAnalysis.id)
        .where(
            FeedbackAnalysis.feedback_id == Feedback.id,
            FeedbackAnalysis.status == "completed",
        )
        .order_by(FeedbackAnalysis.completed_at.desc(), FeedbackAnalysis.id.desc())
        .limit(1)
        .correlate(Feedback)
        .scalar_subquery()
    )


def create_app(
    settings: Settings | None = None,
    predictor_factory: Callable[[], FeedbackPredictor] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    engine = build_engine(settings.database_url)
    upgrade(engine)
    sessions = build_session_factory(engine)
    predictor = (
        predictor_factory()
        if predictor_factory
        else FeedbackPredictor(
            backend=settings.model_backend,
            artifacts_dir=settings.artifacts_dir,
            vncorenlp_dir=settings.vncorenlp_dir,
        )
    )
    with sessions.begin() as session:
        model_ids = _seed_active_models(
            session, predictor.backend_name, settings.artifacts_dir
        )

    app = FastAPI(
        title="Vietnamese Student Feedback Intelligence API",
        version="1.0.0",
    )
    app.state.predictor = predictor
    app.state.engine = engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    get_session = session_dependency(sessions)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(db: Session = Depends(get_session)) -> HealthResponse:
        db.execute(select(1))
        return HealthResponse(
            status="ok", model_backend=predictor.backend_name, database="ok"
        )

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["analysis"])
    def analyze(
        payload: AnalyzeRequest, db: Session = Depends(get_session)
    ) -> AnalyzeResponse:
        started = time.perf_counter()
        try:
            prediction = predictor.predict(payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        latency_ms = round((time.perf_counter() - started) * 1000)
        feedback = Feedback(
            text=prediction.text,
            course_code=payload.course,
            feedback_date=payload.feedback_date,
        )
        db.add(feedback)
        db.flush()
        analysis = _to_analysis(prediction, feedback.id, model_ids, latency_ms)
        db.add(analysis)
        db.commit()
        return _analyze_response(feedback, analysis, predictor.backend_name)

    @app.post("/api/v1/analyze/batch", response_model=BatchResponse, tags=["analysis"])
    async def analyze_batch(
        file: UploadFile = File(...),
        db: Session = Depends(get_session),
    ) -> BatchResponse:
        content = await file.read(settings.max_upload_bytes + 1)
        try:
            rows = parse_feedback_csv(
                content, settings.max_upload_bytes, settings.max_batch_rows
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        now = datetime.now(timezone.utc)
        batch = AnalysisBatch(
            filename=(file.filename or "feedback.csv")[:255],
            source_type="csv",
            total_records=len(rows),
            status="processing",
            started_at=now,
        )
        db.add(batch)
        db.commit()
        try:
            started = time.perf_counter()
            predictions = predictor.predict_many(row.text for row in rows)
            latency_ms = round(
                ((time.perf_counter() - started) * 1000) / len(predictions)
            )
            results: list[tuple[Feedback, FeedbackAnalysis]] = []
            for source_row, (row, prediction) in enumerate(
                zip(rows, predictions), start=2
            ):
                feedback = Feedback(
                    batch_id=batch.id,
                    text=prediction.text,
                    course_code=row.course,
                    feedback_date=row.feedback_date,
                    source_row_number=source_row,
                )
                db.add(feedback)
                db.flush()
                analysis = _to_analysis(
                    prediction, feedback.id, model_ids, latency_ms
                )
                db.add(analysis)
                results.append((feedback, analysis))
            batch.processed_records = len(results)
            batch.status = "completed"
            batch.completed_at = datetime.now(timezone.utc)
            db.commit()
        except Exception as exc:
            db.rollback()
            failed_batch = db.get(AnalysisBatch, batch.id)
            if failed_batch is not None:
                failed_batch.failed_records = failed_batch.total_records
                failed_batch.status = "failed"
                failed_batch.error_message = str(exc)[:2000]
                failed_batch.completed_at = datetime.now(timezone.utc)
                db.commit()
            raise HTTPException(status_code=500, detail="Batch analysis failed") from exc

        return BatchResponse(
            batch_id=batch.id,
            filename=batch.filename or "feedback.csv",
            total_records=batch.total_records,
            processed_records=batch.processed_records,
            failed_records=batch.failed_records,
            status=batch.status,
            results=[
                _feedback_item(feedback, analysis, predictor.backend_name)
                for feedback, analysis in results
            ],
        )

    @app.get("/api/v1/feedback", response_model=FeedbackPage, tags=["feedback"])
    def list_feedback(
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=100),
        sentiment: str | None = None,
        topic: str | None = None,
        course: str | None = None,
        search: str | None = None,
        min_confidence: float | None = Query(default=None, ge=0, le=1),
        date_from: date | None = None,
        date_to: date | None = None,
        db: Session = Depends(get_session),
    ) -> FeedbackPage:
        latest_id = _latest_completed_analysis_id()
        filters = []
        if sentiment:
            filters.append(FeedbackAnalysis.sentiment_label == sentiment)
        if topic:
            filters.append(FeedbackAnalysis.topic_label == topic)
        if course:
            filters.append(Feedback.course_code == course)
        if search:
            filters.append(Feedback.text.ilike(f"%{search}%"))
        if min_confidence is not None:
            filters.extend(
                [
                    FeedbackAnalysis.sentiment_confidence >= min_confidence,
                    FeedbackAnalysis.topic_confidence >= min_confidence,
                ]
            )
        if date_from:
            filters.append(Feedback.feedback_date >= date_from)
        if date_to:
            filters.append(Feedback.feedback_date <= date_to)

        joined = FeedbackAnalysis.id == latest_id
        total = (
            db.scalar(
                select(func.count(Feedback.id))
                .select_from(Feedback)
                .join(FeedbackAnalysis, joined)
                .where(*filters)
            )
            or 0
        )
        rows = db.execute(
            select(Feedback, FeedbackAnalysis, ModelVersion.model_type)
            .join(FeedbackAnalysis, joined)
            .join(ModelVersion, ModelVersion.id == FeedbackAnalysis.sentiment_model_id)
            .where(*filters)
            .order_by(Feedback.created_at.desc(), Feedback.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
        return FeedbackPage(
            items=[
                _feedback_item(feedback, analysis, model_type)
                for feedback, analysis, model_type in rows
            ],
            page=page,
            page_size=page_size,
            total=total,
        )

    @app.get(
        "/api/v1/feedback/{feedback_id}",
        response_model=FeedbackItem,
        tags=["feedback"],
    )
    def get_feedback(
        feedback_id: str, db: Session = Depends(get_session)
    ) -> FeedbackItem:
        latest_id = _latest_completed_analysis_id()
        row = db.execute(
            select(Feedback, FeedbackAnalysis, ModelVersion.model_type)
            .join(FeedbackAnalysis, FeedbackAnalysis.id == latest_id)
            .join(ModelVersion, ModelVersion.id == FeedbackAnalysis.sentiment_model_id)
            .where(Feedback.id == feedback_id)
        ).one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Feedback not found")
        return _feedback_item(*row)

    @app.get(
        "/api/v1/analytics/summary", response_model=SummaryResponse, tags=["analytics"]
    )
    def analytics_summary(
        db: Session = Depends(get_session),
    ) -> SummaryResponse:
        latest_id = _latest_completed_analysis_id()
        latest = (
            select(
                Feedback.id.label("feedback_id"),
                Feedback.course_code,
                FeedbackAnalysis.sentiment_label,
                FeedbackAnalysis.topic_label,
                FeedbackAnalysis.completed_at,
            )
            .join(FeedbackAnalysis, FeedbackAnalysis.id == latest_id)
            .subquery()
        )

        def grouped(column) -> list[CountItem]:
            rows = db.execute(
                select(column, func.count(latest.c.feedback_id))
                .where(column.is_not(None))
                .group_by(column)
                .order_by(func.count(latest.c.feedback_id).desc())
            ).all()
            return [CountItem(label=str(label), count=count) for label, count in rows]

        topic_sentiment_rows = db.execute(
            select(
                latest.c.topic_label,
                latest.c.sentiment_label,
                func.count(latest.c.feedback_id),
            )
            .group_by(latest.c.topic_label, latest.c.sentiment_label)
            .order_by(latest.c.topic_label)
        ).all()
        topic_sentiments: dict[str, dict[str, int]] = {}
        for topic_label, sentiment_label, count in topic_sentiment_rows:
            topic_sentiments.setdefault(str(topic_label), {})[str(sentiment_label)] = count

        trend_rows = db.execute(
            select(
                Feedback.feedback_date,
                FeedbackAnalysis.sentiment_label,
                func.count(Feedback.id),
            )
            .join(FeedbackAnalysis, FeedbackAnalysis.id == latest_id)
            .where(Feedback.feedback_date.is_not(None))
            .group_by(Feedback.feedback_date, FeedbackAnalysis.sentiment_label)
            .order_by(Feedback.feedback_date)
        ).all()
        trends: dict[date, dict[str, int]] = {}
        for feedback_date, sentiment_label, count in trend_rows:
            trends.setdefault(feedback_date, {})[str(sentiment_label)] = count

        return SummaryResponse(
            total_feedback=db.scalar(select(func.count()).select_from(latest)) or 0,
            sentiments=grouped(latest.c.sentiment_label),
            topics=grouped(latest.c.topic_label),
            courses=grouped(latest.c.course_code),
            sentiment_by_topic=[
                TopicSentimentItem(topic=topic, sentiments=sentiments)
                for topic, sentiments in topic_sentiments.items()
            ],
            daily_trend=[
                TrendItem(date=trend_date, sentiments=sentiments)
                for trend_date, sentiments in trends.items()
            ],
            last_analyzed_at=db.scalar(select(func.max(latest.c.completed_at))),
        )

    return app


app = create_app()
