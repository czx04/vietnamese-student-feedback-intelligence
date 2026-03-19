from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from backend.app.config import Settings
from backend.app.database import build_engine, build_session_factory
from backend.app.main import create_app
from backend.app.models import Feedback, FeedbackAnalysis, ModelVersion
from backend.migrations.migration_001_initial_schema import downgrade, upgrade


def test_migration_creates_and_drops_four_tables_and_two_views(tmp_path: Path):
    engine = build_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    upgrade(engine)

    inspector = inspect(engine)
    assert {
        "analysis_batches",
        "feedback",
        "feedback_analyses",
        "model_versions",
    } == set(inspector.get_table_names())
    assert {
        "latest_feedback_analyses",
        "latest_completed_feedback_analyses",
    } == set(inspector.get_view_names())

    downgrade(engine)
    assert inspect(engine).get_table_names() == []
    assert inspect(engine).get_view_names() == []


def test_latest_completed_view_preserves_analysis_history(tmp_path: Path):
    engine = build_engine(f"sqlite:///{tmp_path / 'history.db'}")
    upgrade(engine)
    sessions = build_session_factory(engine)
    now = datetime.now(timezone.utc)
    with sessions.begin() as session:
        sentiment_model = ModelVersion(
            task="sentiment",
            name="test sentiment",
            version="1",
            model_type="test",
            artifact_uri="memory://sentiment",
            labels=["negative", "neutral", "positive"],
            validation_metrics={},
        )
        topic_model = ModelVersion(
            task="topic",
            name="test topic",
            version="1",
            model_type="test",
            artifact_uri="memory://topic",
            labels=["lecturer", "program", "facility", "others"],
            validation_metrics={},
        )
        feedback = Feedback(text="Nội dung đã được phân tích hai lần")
        session.add_all([sentiment_model, topic_model, feedback])
        session.flush()
        common = {
            "feedback_id": feedback.id,
            "sentiment_model_id": sentiment_model.id,
            "topic_model_id": topic_model.id,
            "sentiment_score_type": "uncalibrated_probability",
            "topic_score_type": "uncalibrated_probability",
            "sentiment_scores": {},
            "topic_scores": {},
            "topic_label": "program",
            "topic_confidence": 0.8,
            "status": "completed",
        }
        session.add_all(
            [
                FeedbackAnalysis(
                    **common,
                    sentiment_label="negative",
                    sentiment_confidence=0.7,
                    completed_at=now - timedelta(minutes=1),
                ),
                FeedbackAnalysis(
                    **common,
                    sentiment_label="positive",
                    sentiment_confidence=0.9,
                    completed_at=now,
                ),
            ]
        )

    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT sentiment_label FROM latest_completed_feedback_analyses")
        ).all()
    assert rows == [("positive",)]


def test_confidence_constraint_rejects_invalid_value(tmp_path: Path):
    engine = build_engine(f"sqlite:///{tmp_path / 'constraint.db'}")
    upgrade(engine)
    sessions = build_session_factory(engine)
    with pytest.raises(IntegrityError):
        with sessions.begin() as session:
            sentiment_model = ModelVersion(
                task="sentiment",
                name="s",
                version="1",
                model_type="test",
                artifact_uri="memory://s",
                labels=[],
                validation_metrics={},
            )
            topic_model = ModelVersion(
                task="topic",
                name="t",
                version="1",
                model_type="test",
                artifact_uri="memory://t",
                labels=[],
                validation_metrics={},
            )
            feedback = Feedback(text="invalid")
            session.add_all([sentiment_model, topic_model, feedback])
            session.flush()
            session.add(
                FeedbackAnalysis(
                    feedback_id=feedback.id,
                    sentiment_model_id=sentiment_model.id,
                    topic_model_id=topic_model.id,
                    sentiment_label="positive",
                    sentiment_confidence=1.5,
                    sentiment_score_type="uncalibrated_probability",
                    sentiment_scores={},
                    topic_label="others",
                    topic_confidence=0.5,
                    topic_score_type="uncalibrated_probability",
                    topic_scores={},
                    status="completed",
                )
            )


def test_baseline_models_are_registered_by_task(tmp_path: Path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'seed.db'}")
    app = create_app(settings=settings)
    sessions = build_session_factory(app.state.engine)
    with sessions() as session:
        models = session.scalars(
            select(ModelVersion).where(ModelVersion.is_active.is_(True))
        ).all()
    assert {(model.task, model.model_type) for model in models} == {
        ("sentiment", "linear_svm"),
        ("topic", "linear_svm"),
    }
