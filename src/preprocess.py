"""Preprocess the 2025 StackOverflow Developer Survey into a tidy, model-ready dataset.

Reads data/results.csv (~140MB, gitignored) and writes:
  data/processed.parquet     -- one row per full-time developer, numeric + dummy columns
  data/processed_meta.json   -- column groups, skill choices, top countries
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "results.csv"
OUT_PARQUET = ROOT / "data" / "processed.parquet"
OUT_META = ROOT / "data" / "processed_meta.json"

# (column_name, prefix, top_n_to_keep)  -- top_n=0 keeps all
MULTISELECT_MC = [("DevType", "role", 0)]
MATRIX_COLS = [
    ("Language", "lang", 60),
    ("Database", "db", 25),
    ("Platform", "plat", 20),
    ("Webframe", "web", 15),
    ("DevEnvs", "devenv", 10),
    ("AIModels", "aimodel", 10),
]

JS_MAP = {
    "Very dissatisfied": 1,
    "Slightly dissatisfied": 2,
    "Neither satisfied nor dissatisfied": 3,
    "Slightly satisfied": 4,
    "Very satisfied": 5,
}

SLUG = re.compile(r"[^A-Za-z0-9]+")


def slugify(t: str) -> str:
    return SLUG.sub("_", t).strip("_")


def _norm_token(s: str) -> str:
    s = s.strip()
    if ":" in s:  # Matrix cells may look like "Python:Have; Python:Want"
        s = s.split(":")[0]
    return s.strip()


def _parse_multiselect(series: pd.Series) -> pd.Series:
    out = []
    for v in series.fillna(""):
        if not v:
            out.append([])
        else:
            out.append([_norm_token(t) for t in str(v).split(";") if _norm_token(t)])
    return pd.Series(out, index=series.index)


def _explode(token_lists: pd.Series, prefix: str, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    counts = Counter()
    for lst in token_lists:
        counts.update(lst)
    keep = [k for k, _ in counts.most_common(top_n)] if top_n > 0 else [k for k, _ in counts.most_common()]
    keep = sorted(keep)
    cols = [f"{prefix}_{slugify(k)}" for k in keep]
    M = pd.DataFrame(0, index=token_lists.index, columns=cols, dtype=np.int8)
    name_to_col = {k: c for k, c in zip(keep, cols)}
    for i, lst in token_lists.items():
        for t in lst:
            col = name_to_col.get(t)
            if col:
                M.at[i, col] = 1
    return M, keep


def _years_to_int(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s == "Less than 1 year":
        return 0
    if s == "More than 50 years":
        return 51
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return np.nan


def build_processed() -> pd.DataFrame:
    print(f"Loading {RAW} ...")
    df = pd.read_csv(RAW, low_memory=False)
    print(f"  raw shape: {df.shape}")

    # Filter to professional developers currently in work
    # (Module 1: data cleansing). 2025 simplified the Employment value to
    # "Employed" so we keep all working developers (incl. part-time /
    # independent) and gate on MainBranch to ensure they self-identify as
    # developers by profession.
    emp = df["Employment"].fillna("")
    branch = df["MainBranch"].fillna("")
    working = emp.str.contains(r"Employed|Independent|freelancer|self-employed",
                                case=False, regex=True)
    is_dev = branch.str.contains("developer by profession", case=False)
    df = df[working & is_dev].copy()
    print(f"  working developers: {df.shape}")

    # Numeric conversions
    df["WorkExp"] = df["WorkExp"].apply(_years_to_int)
    df["YearsCode"] = df["YearsCode"].apply(_years_to_int)
    df["CompTotal"] = pd.to_numeric(df["CompTotal"], errors="coerce")
    df["log_comp"] = np.log1p(df["CompTotal"])

    # JobSat -> numeric & binary high (1-5; high = 4 or 5)
    df["JobSat_num"] = df["JobSat"].map(JS_MAP)
    df["JobSat_high"] = (df["JobSat_num"] >= 4).astype("Int64")

    # AI usage
    sel = df["AISelect"].fillna("")
    df["ai_user"] = sel.str.contains(r"\bYes\b", case=False, regex=True).astype(int)
    df["ai_user"] = ((sel != "") & (df["ai_user"] == 1)).astype(int)

    # Explode multi-select + matrix columns
    parts: dict[str, pd.DataFrame] = {}
    labels: dict[str, dict] = {}
    for col, pref, n in MULTISELECT_MC + MATRIX_COLS:
        if col not in df.columns:
            continue
        tl = _parse_multiselect(df[col])
        M, keep = _explode(tl, pref, n)
        parts[col] = M
        labels[col] = {"prefix": pref, "kept": keep, "n_cols": int(M.shape[1])}

    # Top countries -> dummies
    top_countries = df["Country"].value_counts().head(10).index.tolist()
    cmat = pd.DataFrame(0, index=df.index,
                        columns=[f"country_{slugify(c)}" for c in top_countries], dtype=np.int8)
    for c, col in zip(top_countries, cmat.columns):
        cmat[col] = (df["Country"] == c).astype(int)
    parts["country"] = cmat
    labels["country"] = {"prefix": "country", "kept": top_countries, "n_cols": int(cmat.shape[1])}

    # Base frame
    id_col = "ResponseId" if "ResponseId" in df.columns else df.columns[0]
    base = pd.DataFrame({
        "id": df[id_col].values,
        "Country": df["Country"].values,
        "RemoteWork": df["RemoteWork"].values,
        "EdLevel": df["EdLevel"].values,
        "WorkExp": df["WorkExp"].values,
        "YearsCode": df["YearsCode"].values,
        "CompTotal": df["CompTotal"].values,
        "log_comp": df["log_comp"].values,
        "JobSat_num": df["JobSat_num"].values,
        "JobSat_high": df["JobSat_high"].values,
        "ai_user": df["ai_user"].values,
        "AISent": df["AISent"].values,
        "AIThreat": df["AIThreat"].values,
    })

    out = pd.concat([base.reset_index(drop=True)] + [p.reset_index(drop=True) for p in parts.values()],
                    axis=1)

    # IQR outlier flag on log_comp (don't drop -- let analyses filter)
    q1, q3 = out["log_comp"].quantile([0.25, 0.75])
    iqr = q3 - q1
    out["comp_inlier"] = ((out["log_comp"] >= q1 - 3 * iqr) & (out["log_comp"] <= q3 + 3 * iqr)).astype(int)

    # Drop rows missing the essentials
    out = out.dropna(subset=["WorkExp", "log_comp", "JobSat_num"]).reset_index(drop=True)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    meta = {
        "shape": list(out.shape),
        "skill_groups": labels,
        "top_countries": top_countries,
        "all_columns": list(out.columns),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  saved {out.shape} -> {OUT_PARQUET}")
    return out


if __name__ == "__main__":
    build_processed()
