"""A 2-D map of skill space for the app's sky chart.

Each skill is placed by how often developers use it together with every
other skill: the distance between two skills is 1 - Jaccard similarity of
the sets of developers who know them, embedded in two dimensions with
t-SNE (precomputed metric, fixed seed). Skills that are usually learned
together sit near each other, so a developer's stack forms a compact
"constellation" and the skills its peers hold sit close by.

Writes data/skill_map.json: {"skills": {column: [x, y]}, ...} with x and y
scaled to [-1, 1].

Run:  .venv/bin/python -m src.skymap
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed.parquet"
OUT_JSON = ROOT / "data" / "skill_map.json"
SKILL_PREFIXES = ("lang_", "db_", "plat_", "web_", "devenv_", "aimodel_")
PERPLEXITY = 12
SEED = 42


def jaccard_distance(X: np.ndarray) -> np.ndarray:
    """1 - |A n B| / |A u B| between the columns of a binary matrix."""
    X = X.astype(np.float64)
    inter = X.T @ X
    counts = np.diag(inter)
    union = counts[:, None] + counts[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    return d


def run() -> dict:
    df = pd.read_parquet(PROCESSED)
    skills = [c for c in df.columns if c.startswith(SKILL_PREFIXES)]
    D = jaccard_distance(df[skills].to_numpy())
    xy = TSNE(n_components=2, metric="precomputed", init="random",
              perplexity=PERPLEXITY, random_state=SEED).fit_transform(D)
    xy = xy - xy.mean(axis=0)
    xy = xy / np.abs(xy).max()
    out = {
        "method": f"t-SNE (perplexity={PERPLEXITY}, seed={SEED}) on 1 - Jaccard "
                  "similarity of the developers who know each pair of skills",
        "n_developers": int(len(df)),
        "skills": {s: [round(float(x), 4), round(float(y), 4)] for s, (x, y) in zip(skills, xy)},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"saved {len(skills)} skill positions -> {OUT_JSON}")
    return out


if __name__ == "__main__":
    run()
