from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ml.data import ROOT


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'feedback.db'}"
    )
    model_backend: str = os.getenv("MODEL_BACKEND", "baseline")
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))
    vncorenlp_dir: Path = Path(
        os.getenv("VNCORENLP_DIR", str(ROOT / "artifacts" / "vncorenlp"))
    )
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
    max_batch_rows: int = int(os.getenv("MAX_BATCH_ROWS", "5000"))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    )
