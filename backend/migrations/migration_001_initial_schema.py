from __future__ import annotations

import argparse

from sqlalchemy import inspect, text

from backend.app import models  # noqa: F401 - registers ORM tables
from backend.app.config import Settings
from backend.app.database import Base, build_engine


LATEST_VIEW_SQL = """
CREATE VIEW latest_feedback_analyses AS
SELECT * FROM (
    SELECT feedback_analyses.*,
           ROW_NUMBER() OVER (
               PARTITION BY feedback_id
               ORDER BY created_at DESC, id DESC
           ) AS row_number
    FROM feedback_analyses
) ranked
WHERE row_number = 1
"""

LATEST_COMPLETED_VIEW_SQL = """
CREATE VIEW latest_completed_feedback_analyses AS
SELECT * FROM (
    SELECT feedback_analyses.*,
           ROW_NUMBER() OVER (
               PARTITION BY feedback_id
               ORDER BY completed_at DESC, id DESC
           ) AS row_number
    FROM feedback_analyses
    WHERE status = 'completed'
) ranked
WHERE row_number = 1
"""


def upgrade(engine) -> None:
    Base.metadata.create_all(engine)
    existing = set(inspect(engine).get_view_names())
    with engine.begin() as connection:
        if "latest_feedback_analyses" not in existing:
            connection.execute(text(LATEST_VIEW_SQL))
        if "latest_completed_feedback_analyses" not in existing:
            connection.execute(text(LATEST_COMPLETED_VIEW_SQL))


def downgrade(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("DROP VIEW IF EXISTS latest_completed_feedback_analyses"))
        connection.execute(text("DROP VIEW IF EXISTS latest_feedback_analyses"))
    Base.metadata.drop_all(engine)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply or revert migration 001")
    parser.add_argument("direction", choices=("up", "down"))
    parser.add_argument("--database-url", default=Settings().database_url)
    args = parser.parse_args()
    engine = build_engine(args.database_url)
    (upgrade if args.direction == "up" else downgrade)(engine)
    print(f"migration 001: {args.direction} complete")


if __name__ == "__main__":
    main()
