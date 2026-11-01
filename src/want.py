"""Load the 2025 'Want to work with' columns as per-respondent skill sets.

These are separate from the HaveWorkedWith columns we use elsewhere. The
recommender uses them to boost skills that the user's peers *want* to learn
(a real demand signal).

The pipeline parses the relevant columns of data/results.csv (~135 MB,
gitignored) once and saves the result to data/want_sets.parquet (one row
per respondent-skill pair, small enough to commit). The app and the
recommender load that extract, so a deployment does not need the raw file.

Run:  .venv/bin/python -m src.want     (rebuilds the extract)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import skill_column_names

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "results.csv"
PROCESSED = ROOT / "data" / "processed.parquet"
META = ROOT / "data" / "processed_meta.json"
EXTRACT = ROOT / "data" / "want_sets.parquet"

WANT_COLS = [
    "LanguageWantToWorkWith",
    "DatabaseWantToWorkWith",
    "PlatformWantToWorkWith",
    "WebframeWantToWorkWith",
    "DevEnvsWantToWorkWith",
    "AIModelsWantToWorkWith",
]
# Each "Want" column is mapped through the kept answers of its matching
# "Have" column, so a wanted skill lands on the same column the
# preprocessed skill matrix uses.
HAVE_BY_WANT = {
    "LanguageWantToWorkWith": "LanguageHaveWorkedWith",
    "DatabaseWantToWorkWith": "DatabaseHaveWorkedWith",
    "PlatformWantToWorkWith": "PlatformHaveWorkedWith",
    "WebframeWantToWorkWith": "WebframeHaveWorkedWith",
    "DevEnvsWantToWorkWith": "DevEnvsHaveWorkedWith",
    "AIModelsWantToWorkWith": "AIModelsHaveWorkedWith",
}

_cached = None


def _column_maps() -> dict[str, dict[str, str]]:
    """{want column: {survey answer: skill column}} from processed_meta.json."""
    groups = json.loads(META.read_text())["skill_groups"]
    return {
        want: skill_column_names(sorted(groups[have]["kept"]), groups[have]["prefix"])
        for want, have in HAVE_BY_WANT.items()
    }


def load_want_sets() -> dict[str, set[str]]:
    """Return {ResponseId: set(skill column names the respondent wants)}.

    Reads data/want_sets.parquet; builds it from the raw survey first if it
    is missing.
    """
    global _cached
    if _cached is not None:
        return _cached
    if not EXTRACT.exists():
        build_extract()
    long = pd.read_parquet(EXTRACT)
    out: dict[str, set[str]] = {}
    for rid, skill in zip(long["id"].astype(str), long["skill"]):
        out.setdefault(rid, set()).add(skill)
    _cached = out
    return out


def build_extract() -> Path:
    """Parse the raw WantToWorkWith answers and save data/want_sets.parquet."""
    pairs = _parse_raw()
    long = pd.DataFrame(sorted((rid, s) for rid, skills in pairs.items() for s in skills),
                        columns=["id", "skill"])
    long.to_parquet(EXTRACT, index=False)
    print(f"saved {len(long):,} respondent-skill pairs for "
          f"{long['id'].nunique():,} respondents -> {EXTRACT}")
    return EXTRACT


def _parse_raw() -> dict[str, set[str]]:
    """{ResponseId: wanted skill columns} from data/results.csv.

    Covers every skill column in processed.parquet. Answers that were not
    kept as a column (the long tail) are dropped.
    """
    maps = _column_maps()

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
            name_to_col = maps[col]
            for token in (t.strip() for t in cell.split(";")):
                if ":" in token:
                    token = token.split(":")[0].strip()
                skill = name_to_col.get(token)
                if skill:
                    skills.add(skill)
        out[row["ResponseId"]] = skills
    return out


if __name__ == "__main__":
    build_extract()
