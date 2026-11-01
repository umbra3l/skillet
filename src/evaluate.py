"""Offline evaluation of the recommender and its score weights.

Two held-out tasks on real respondents. Each test respondent is removed
from their own peer pool (`exclude_row`), and recommends from their own
role / country / experience / AI-tool use, exactly as the app would.

  A. Hidden-skill recovery: hide two of the respondent's skills (chosen
     among skills the recommender can recommend: positive salary ROI) and
     ask for the top 5 from the rest of their stack.
     hit@5    = at least one hidden skill is in the top 5
     recall@5 = share of the two hidden skills found
  B. Wanted skills: from the full stack, what share of the top 5 the
     respondent says they *want* to work with (WantToWorkWith) --
     precision@5. This is the closest the survey gets to "what should
     this developer learn next".

Each method is also scored on value: the mean stand-alone salary effect
(%) of the skills it recommends.

Methods: a popularity baseline (most common skills in the survey), a
salary-only baseline (highest ROI), the KNN with prevalence only, the
current WEIGHTS, and a grid over the ROI and demand weights. Weights are
chosen on a validation half and reported on a separate test half, with
bootstrap 95% intervals.

Writes data/recommender_eval.json.

Run:  .venv/bin/python -m src.evaluate
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommend import WEIGHTS, SkillRadar
from src.want import load_want_sets

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "recommender_eval.json"

N_USERS = 1000          # split 50/50 into validation and test
N_HIDDEN = 2
TOP_N = 5
SEED = 42
ROI_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
WANT_GRID = (0.0, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0)
ZERO = {"roi": 0.0, "want": 0.0, "sig": 0.0, "ai": 0.0}


def _candidates(sr: SkillRadar, stack: list[str], profile: dict, row: int) -> pd.DataFrame:
    """Every candidate the recommender would score, with its components.

    The candidate set (peer prevalence > 2%, positive ROI) does not depend
    on the weights, so one call with zero weights returns all of them and
    any weighting can be re-ranked offline.
    """
    out = sr.recommend(stack, top_n=len(sr.skills), weights=ZERO, exclude_row=row, **profile)
    recs = out["recommendations"]
    return pd.DataFrame({
        "skill": [r["skill"] for r in recs],
        "prev": [r["nbr_prevalence_pct"] / 100 for r in recs],
        "want": [r["nbr_want_pct"] / 100 for r in recs],
    })


def _rank(c: pd.DataFrame, roi: pd.DataFrame, w: dict, ai_user: bool) -> list[str]:
    if c.empty:
        return []
    coef = roi.loc[c["skill"], "coef_log_comp"].to_numpy()
    sig = (roi.loc[c["skill"], "p_value"].to_numpy() < 0.05).astype(float)
    is_ai = c["skill"].str.startswith("aimodel_").to_numpy() & bool(ai_user)
    score = (c["prev"].to_numpy() + w["roi"] * coef + w["want"] * c["want"].to_numpy()
             + w["sig"] * sig + w["ai"] * is_ai)
    return list(c["skill"].to_numpy()[np.argsort(-score, kind="stable")][:TOP_N])


def _static_rank(order: list[str], have: set[str]) -> list[str]:
    return [s for s in order if s not in have][:TOP_N]


def _boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> list[float]:
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run() -> dict:
    rng = np.random.default_rng(SEED)
    sr = SkillRadar()
    df = sr._data()
    roi = sr.roi.set_index("skill")
    want = load_want_sets()
    eligible = set(roi.index[roi["coef_log_comp"] > 0]) & set(sr.skills)
    roles = [c for c in df.columns if c.startswith("role_")]
    pct = roi["pct_salary_impact"]

    # Baseline orderings (no personalisation beyond excluding owned skills)
    popularity = list(df[sorted(eligible)].mean().sort_values(ascending=False).index)
    roi_only = list(roi.loc[sorted(eligible), "coef_log_comp"].sort_values(ascending=False).index)

    # Test respondents: >= 6 skills, >= N_HIDDEN hideable ones, and at least
    # one wanted skill they do not have that the recommender could suggest.
    X = df[sr.skills].to_numpy()
    pool = []
    for i in rng.permutation(len(df)):
        have = {s for s, v in zip(sr.skills, X[i]) if v}
        wanted = (want.get(str(df.at[i, "id"]), set()) - have) & eligible
        if len(have) >= 6 and len(have & eligible) >= N_HIDDEN and wanted:
            pool.append((int(i), have, wanted))
        if len(pool) == N_USERS:
            break

    methods = {"popularity": None, "salary_only": None, "knn_prevalence_only": ZERO,
               "current_weights": WEIGHTS}
    grid = {f"roi={r},want={w}": {**WEIGHTS, "roi": r, "want": w}
            for r, w in itertools.product(ROI_GRID, WANT_GRID)}

    rows = []  # one per (user, method)
    for n, (i, have, wanted) in enumerate(pool):
        r = df.iloc[i]
        profile = dict(role=next(c for c in roles if r[c] == 1), country=r["Country"],
                       workexp=int(r["WorkExp"]), ai_user=bool(r["ai_user"]))
        hidden = set(rng.choice(sorted(have & eligible), size=N_HIDDEN, replace=False))
        stack_a = sorted(have - hidden)
        cand_a = _candidates(sr, stack_a, profile, i)
        cand_b = _candidates(sr, sorted(have), profile, i)
        split = "validation" if n % 2 == 0 else "test"

        def score(name: str, top_a: list[str], top_b: list[str]) -> None:
            found = len(set(top_a) & hidden)
            rows.append({
                "user": n, "split": split, "method": name,
                "hit_at_5": float(found > 0), "recall_at_5": found / N_HIDDEN,
                "want_precision_at_5": len(set(top_b) & wanted) / TOP_N,
                "mean_salary_effect_pct": float(pct.loc[top_b].mean()) if top_b else np.nan,
            })

        score("popularity", _static_rank(popularity, set(stack_a)), _static_rank(popularity, have))
        score("salary_only", _static_rank(roi_only, set(stack_a)), _static_rank(roi_only, have))
        for name, w in {**{k: v for k, v in methods.items() if v is not None}, **grid}.items():
            score(name, _rank(cand_a, roi, w, profile["ai_user"]),
                  _rank(cand_b, roi, w, profile["ai_user"]))
        if (n + 1) % 200 == 0:
            print(f"  evaluated {n + 1}/{len(pool)} respondents")

    res = pd.DataFrame(rows)
    metrics = ["hit_at_5", "recall_at_5", "want_precision_at_5", "mean_salary_effect_pct"]
    agg = res.groupby(["split", "method"])[metrics].mean()

    # Weight selection on validation: the most accurate grid point on the
    # primary task (B) that keeps at least the current weights' salary value.
    val = agg.loc["validation"]
    floor = val.loc["current_weights", "mean_salary_effect_pct"]
    ok = val.loc[[g for g in grid if val.loc[g, "mean_salary_effect_pct"] >= floor]]
    chosen = ok["want_precision_at_5"].idxmax()

    test = res[res["split"] == "test"]
    report = {}
    for name in ["popularity", "salary_only", "knn_prevalence_only", "current_weights", chosen]:
        t = test[test["method"] == name]
        report[name] = {m: {"mean": float(t[m].mean()), "ci95": _boot_ci(t[m].to_numpy(), rng)}
                        for m in metrics}

    # Paired comparison on the same test users: best grid point minus current.
    by = test.set_index(["method", "user"])
    paired = {}
    for m in metrics:
        d = (by.loc[chosen, m] - by.loc["current_weights", m]).dropna().to_numpy()
        paired[m] = {"mean_diff": float(d.mean()), "ci95": _boot_ci(d, rng)}

    out = {
        "n_users": {"validation": int((res.split == "validation").sum() / res.method.nunique()),
                    "test": int((res.split == "test").sum() / res.method.nunique())},
        "n_hidden": N_HIDDEN, "top_n": TOP_N,
        "current_weights": WEIGHTS,
        # The best grid point is reported, not adopted: on test it trades hit
        # rate for want-precision and salary value (see the paired block), so
        # WEIGHTS in src/recommend.py stays as it is. REPORT.md section 9.7.
        "best_grid_point_on_validation": {
            "name": chosen, "weights": grid[chosen], "adopted": False,
            "rule": "max want_precision_at_5 with mean salary effect >= current "
                    "weights (validation)"},
        "test": report,
        "test_paired_best_minus_current": paired,
        "validation_grid": val.loc[list(grid)].round(4).reset_index().to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"\nTest set (n={out['n_users']['test']}), 95% bootstrap CI:")
    print(f"  {'method':24s} {'hit@5':>16s} {'want-prec@5':>16s} {'salary effect':>18s}")
    for name, m in report.items():
        f = lambda k, d=3: f"{m[k]['mean']:.{d}f} [{m[k]['ci95'][0]:.{d}f},{m[k]['ci95'][1]:.{d}f}]"
        print(f"  {name:24s} {f('hit_at_5'):>16s} {f('want_precision_at_5'):>16s} "
              f"{f('mean_salary_effect_pct', 1):>18s}")
    print(f"best grid point on validation (reported, not adopted): {chosen}")
    print("paired difference on test (best grid point - current), 95% bootstrap CI:")
    for m, v in paired.items():
        print(f"  {m:24s} {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print(f"saved -> {OUT_JSON}")
    return out


if __name__ == "__main__":
    run()
