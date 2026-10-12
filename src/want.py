"""Load the 2025 'Want to work with' columns as per-respondent skill sets.

These are separate from the HaveWorkedWith columns we use elsewhere. The
recommender uses them to boost skills that the user's peers *want* to learn
(a real demand signal).

This module reads ONLY the relevant columns from data/results.csv to keep
memory low (~135 MB total but filtered to a few columns + the response ids
in processed.parquet).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import SLUG

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "results.csv"
PROCESSED = ROOT / "data" / "processed.parquet"

WANT_COLS = [
    "LanguageWantToWorkWith",
    "DatabaseWantToWorkWith",
    "PlatformWantToWorkWith",
    "WebframeWantToWorkWith",
    "DevEnvsWantToWorkWith",
    "AIModelsWantToWorkWith",
]
PREFIX_BY_COL = {
    "LanguageWantToWorkWith": "lang",
    "DatabaseWantToWorkWith": "db",
    "PlatformWantToWorkWith": "plat",
    "WebframeWantToWorkWith": "web",
    "DevEnvsWantToWorkWith": "devenv",
    "AIModelsWantToWorkWith": "aimodel",
}

_cached = None


def _slug(t: str) -> str:
    return SLUG.sub("_", t).strip("_")


def load_want_sets() -> dict[str, set[str]]:
    """Return {ResponseId: set(slugged skill names in cluster space)}.

    Only includes skills that exist in the cluster-space skill list (loaded
    from the cluster_model.joblib bundle), so downstream code can intersect
    cleanly.
    """
    global _cached
    if _cached is not None:
        return _cached

    import joblib
    cluster_skills = set(joblib.load(ROOT / "data" / "cluster_model.joblib")["skills"])
    prefixes = tuple(PREFIX_BY_COL.values())

    keep_ids = set(pd.read_parquet(PROCESSED, columns=["id"])["id"].astype(str))
    usecols = ["ResponseId"] + WANT_COLS
    raw = pd.read_csv(RAW, usecols=usecols, low_memory=False, encoding="utf-8-sig")
    raw["ResponseId"] = raw["ResponseId"].astype(str)
    raw = raw[raw["ResponseId"].isin(keep_ids)]

    out: dict[str, set[str]] = {}
    for _, row in raw.iterrows():
        skills: set[str] = set()
        for col in WANT_COLS:
            cell = row.get(col)
            if not isinstance(cell, str) or not cell:
                continue
            pref = PREFIX_BY_COL[col]
            for token in (t.strip() for t in cell.split(";")):
                if not token:
                    continue
                if ":" in token:
                    token = token.split(":")[0]
                slug = f"{pref}_{_slug(token)}"
                if slug in cluster_skills:
                    skills.add(slug)
        out[row["ResponseId"]] = skills
    _cached = out
    return out
