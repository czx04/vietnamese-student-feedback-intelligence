"""Word-segment UIT-VSFC text once and cache it for PhoBERT."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ml.data import RAW_DIR, SPLITS, load_split, prepare_training_frame, validate_labels
from ml.phobert_common import SEGMENTED_DIR, write_json


def segment_text(segmenter: Any, text: str) -> str:
    segmented = " ".join(segmenter.word_segment(text)).strip()
    if not segmented:
        raise ValueError(f"VnCoreNLP returned empty text for input: {text!r}")
    return segmented


def build_segmenter(vncorenlp_dir: Path, download_model: bool = False) -> Any:
    if shutil.which("java") is None:
        raise RuntimeError("Java is required by VnCoreNLP but was not found in PATH")

    try:
        import py_vncorenlp
    except ImportError as exc:
        raise RuntimeError(
            "py-vncorenlp is not installed. Run "
            "`python -m pip install -r requirements-phobert.txt`."
        ) from exc

    vncorenlp_dir.mkdir(parents=True, exist_ok=True)
    jar_exists = any(vncorenlp_dir.rglob("VnCoreNLP*.jar"))
    if not jar_exists:
        if not download_model:
            raise FileNotFoundError(
                f"VnCoreNLP model is missing from {vncorenlp_dir}. Re-run with "
                "--download-model."
            )
        py_vncorenlp.download_model(save_dir=str(vncorenlp_dir))

    return py_vncorenlp.VnCoreNLP(
        annotators=["wseg"],
        save_dir=str(vncorenlp_dir),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_split(
    split: str,
    segmenter: Any,
    output_dir: Path = SEGMENTED_DIR,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    output_path = output_dir / f"{split}.csv"
    if output_path.exists() and not force:
        cached = pd.read_csv(output_path)
        return output_path, {
            "rows": len(cached),
            "cached": True,
            "preprocessing": {},
        }

    frame = load_split(split)
    preprocessing: dict[str, int] = {}
    if split == "train":
        frame, preprocessing = prepare_training_frame(frame)
    frame["phobert_text"] = [
        segment_text(segmenter, text) for text in frame["text"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path, {
        "rows": len(frame),
        "cached": False,
        "preprocessing": preprocessing,
    }


def prepare_data(
    splits: Iterable[str],
    vncorenlp_dir: Path,
    output_dir: Path = SEGMENTED_DIR,
    download_model: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    selected = list(dict.fromkeys(splits))
    invalid = set(selected).difference(SPLITS)
    if invalid:
        raise ValueError(f"Unknown splits: {sorted(invalid)}")

    # Check the label contract before spending time on segmentation.
    validate_labels({split: load_split(split) for split in selected})
    segmenter = build_segmenter(vncorenlp_dir, download_model=download_model)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vncorenlp_dir": str(vncorenlp_dir.resolve()),
        "splits": {},
    }
    for split in selected:
        path, summary = prepare_split(
            split,
            segmenter=segmenter,
            output_dir=output_dir,
            force=force,
        )
        summary["path"] = str(path.resolve())
        summary["raw_sha256"] = _sha256(RAW_DIR / f"{split}.json")
        manifest["splits"][split] = summary
        print(f"{split}: {summary['rows']} rows -> {path}")

    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache VnCoreNLP word segmentation for PhoBERT."
    )
    parser.add_argument(
        "--splits", nargs="+", choices=SPLITS, default=list(SPLITS)
    )
    parser.add_argument(
        "--vncorenlp-dir", type=Path, default=Path("artifacts/vncorenlp")
    )
    parser.add_argument("--output-dir", type=Path, default=SEGMENTED_DIR)
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the VnCoreNLP model if it is not cached.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing CSV files."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_data(
        splits=args.splits,
        vncorenlp_dir=args.vncorenlp_dir,
        output_dir=args.output_dir,
        download_model=args.download_model,
        force=args.force,
    )


if __name__ == "__main__":
    main()
