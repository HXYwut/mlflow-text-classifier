"""Data pipeline: build v1 (raw) and v2 (cleaned) datasets from 20 Newsgroups.

v1: keep headers, footers, quotes  (noisy, like a real inbox)
v2: strip headers, footers, quotes  (clean, content-only)

Both versions are saved as CSV under ``data/raw/v1`` and ``data/raw/v2`` and
tracked by DVC (only the .dvc pointer files live in git).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = ["rec.sport.hockey", "sci.space"]
LABEL_MAP = {"rec.sport.hockey": 0, "sci.space": 1}

ROOT = Path(__file__).resolve().parent.parent
V1_DIR = ROOT / "data" / "raw" / "v1"
V2_DIR = ROOT / "data" / "raw" / "v2"


def _build(split: str, remove: tuple[str, ...]) -> pd.DataFrame:
    bunch = fetch_20newsgroups(
        subset=split,
        categories=CATEGORIES,
        remove=remove,
        random_state=42,
    )
    # Note: because CATEGORIES is ordered ["rec.sport.hockey", "sci.space"],
    # bunch.target is already 0=hockey, 1=space.
    return pd.DataFrame({"text": bunch.data, "label": bunch.target})


def build_all() -> None:
    # v1: raw (headers/footers/quotes kept)
    V1_DIR.mkdir(parents=True, exist_ok=True)
    _build("train", remove=()).to_csv(V1_DIR / "train.csv", index=False)
    _build("test", remove=()).to_csv(V1_DIR / "test.csv", index=False)

    # v2: cleaned
    V2_DIR.mkdir(parents=True, exist_ok=True)
    _build("train", remove=("headers", "footers", "quotes")).to_csv(V2_DIR / "train.csv", index=False)
    _build("test", remove=("headers", "footers", "quotes")).to_csv(V2_DIR / "test.csv", index=False)

    print(f"[data] v1 -> {V1_DIR}")
    print(f"[data] v2 -> {V2_DIR}")


if __name__ == "__main__":
    build_all()
