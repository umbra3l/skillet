"""KNN-based upskilling recommender.

  1. Build the user's binary skill vector in the 86-skill cluster space.
  2. Find the k nearest developers in the survey by Jaccard distance on
     the binary skill matrix (KNN -- Module 4). The KNN is built once on
     the full dataset; the recommender then intersects the neighbour pool
     with an optional country / workexp mask (sidebar inputs) so the peer
     set reflects the visitor's context.
  3. For every skill the user does NOT have, compute
        score(s) =  peer_prevalence(s)                   # how common in your filtered peers
                + 0.5 * salary_ROI(s)                   # per-skill MLR coefficient (log pts)
                + 0.2 * max(peer_salary_premium, 0)     # how much your peer group earns
                + 0.3 * peer_demand(s)                  # share of peers who "want" to learn it
                + 0.02 if ai_user and s starts with "aimodel_"   # small nudge
                + 0.05 if p(s) < 0.05                   # significance bonus
     and rank. The salary-ROI coefficient is the partial effect from
     log_comp ~ work_exp + country + skill, so it is already conditioned
     on work experience and country; the sidebar inputs additionally
     shape *which peers* the score uses.
  4. Cold-start (no skills listed): the same salary-ROI prior, with the
     global skill prevalence (within the same country/experience filter)
     and the +0.05 significance bonus. No peer_prevalence, peer_salary
     premium, or peer_demand terms exist in cold start -- there is no
     peer group yet.
  5. Return the top N with the per-skill evidence strings. The
     satisfaction_coef key is None for skills the satisfaction model
     never scored (it was trained on the top-30 most common skills); the
     app renders "not modeled" rather than 0.000 for those.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class SkillRadar:
    def __init__(self) -> None:
        cluster = joblib.load(DATA / "cluster_model.joblib")
        self.skills: list[str] = cluster["skills"]  # the 86 cluster-space skills
        self.personas: dict[int, dict] = {
            p["cluster"]: p
            for p in json.loads((DATA / "cluster_meta.json").read_text())["personas"]
        }
        self.roi = pd.read_csv(DATA / "skill_roi.csv")
        self.roi_map: dict[str, dict] = {r["skill"]: r.to_dict() for _, r in self.roi.iterrows()}
        sat = pd.read_csv(DATA / "satisfaction_coefs.csv")
        self.sat_map: dict[str, float] = dict(zip(sat["feature"], sat["coef"]))
        self._df: pd.DataFrame | None = None
        self._nn: NearestNeighbors | None = None
        self._nn_X: np.ndarray | None = None
        self._want: dict[str, set[str]] | None = None

    # --- data accessors --------------------------------------------------------
    def _data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_parquet(DATA / "processed.parquet")
        return self._df

    def _skill_matrix(self) -> np.ndarray:
        X = self._nn_X
        if X is None:
            X = self._data()[self.skills].astype(float).values
            self._nn_X = X
        return X

    # --- persona (overlap-based, for display) ---------------------------------
    def persona_for(self, current_skills: Iterable[str]) -> int:
        have = {s for s in current_skills}
        best, best_score = None, -1.0
        for cid, p in self.personas.items():
            score = sum(ts["lift"] for ts in p["top_skills"] if ts["skill"] in have)
            score += 1e-6 * p["n"]  # deterministic tie-break
            if score > best_score:
                best, best_score = cid, score
        if best is None or best_score <= 0:
            best = max(self.personas, key=lambda c: self.personas[c]["n"])
        return int(best)

    # --- KNN recommender -------------------------------------------------------
    def recommend(self, current_skills: Iterable[str], top_n: int = 5,
                  k_neighbors: int = 300,
                  country: str | None = None,
                  workexp: int | None = None,
                  role: str | None = None,
                  ai_user: bool | None = None) -> dict:
        df_full = self._data()
        have = {s for s in current_skills}
        persona_id = self.persona_for(current_skills)
        persona = self.personas.get(persona_id, {})

        # Build a mask for the optional sidebar constraints (country,
        # workexp +/- 10 years) and apply it to the full df. The KNN
        # is built once on the full df; we query it for a generous
        # number of neighbours and intersect with the mask.
        full_idx = np.arange(len(df_full))
        mask = np.ones(len(df_full), dtype=bool)
        if country is not None:
            mask &= (df_full["Country"].values == country)
        if workexp is not None:
            lo, hi = max(0, int(workexp) - 10), int(workexp) + 10
            mask &= (df_full["WorkExp"].values >= lo) & (df_full["WorkExp"].values <= hi)
        valid_full_idx = set(full_idx[mask].tolist())
        # If the filter collapses the pool, fall back to the full df.
        use_mask = len(valid_full_idx) >= 50

        # Cold-start: no skills listed -> recommend globally high-ROI,
        # high-prevalence skills within the country/experience filter.
        if not have:
            df = df_full[mask] if use_mask else df_full
            return self._cold_start(top_n, df)

        skills = self.skills
        v = np.array([1.0 if s in have else 0.0 for s in skills], dtype=float).reshape(1, -1)
        X = self._skill_matrix()
        if self._nn is None:
            self._nn = NearestNeighbors(
                n_neighbors=min(5000, X.shape[0]),
                metric="jaccard", n_jobs=-1,
            ).fit(X)
        # Query the KNN for every neighbour it knows, then keep only
        # those that pass the country/workexp mask. This makes the sidebar
        # inputs change the peer set (and hence the recommendations).
        dists, idxs = self._nn.kneighbors(v, n_neighbors=self._nn.n_neighbors)
        keep = np.isfinite(dists[0])
        cand = idxs[0][keep]
        if use_mask:
            cand = np.array([i for i in cand if int(i) in valid_full_idx],
                            dtype=cand.dtype)
        if cand.size == 0:
            return self._cold_start(top_n,
                                    df_full[mask] if use_mask else df_full)
        # Take the top k_neighbors from the filtered set.
        cand = cand[:k_neighbors]
        neighbours = df_full.iloc[cand]
        nbr_log_comp = neighbours["log_comp"].mean()
        overall_log_comp = df_full["log_comp"].mean()
        salary_premium = float(nbr_log_comp - overall_log_comp)

        # WantToWorkWith demand signal (lazy-loaded; first call reads the raw CSV)
        if self._want is None:
            from src.want import load_want_sets
            self._want = load_want_sets()
        nbr_ids = neighbours["id"].astype(str).tolist()
        want_counts: dict[str, int] = {s: 0 for s in skills}
        for nid in nbr_ids:
            ws = self._want.get(nid, set())
            for s in ws:
                if s in want_counts:
                    want_counts[s] += 1
        n_nbr = max(len(nbr_ids), 1)

        cand: dict[str, float] = {}
        for s in skills:
            if s in have:
                continue
            freq = float(neighbours[s].mean())
            if freq <= 0.02:
                continue
            roi = self.roi_map.get(s, {})
            roi_coef = float(roi.get("coef_log_comp", 0.0))
            # Only recommend skills associated with higher compensation.
            # A negative-ROI recommendation is an honest but counter-productive
            # product signal; surface it in the coefficient chart instead.
            if roi_coef <= 0:
                continue
            p_val = float(roi.get("p_value", 1.0))
            want_frac = want_counts[s] / n_nbr
            score = freq + 0.5 * roi_coef + 0.2 * max(salary_premium, 0.0) + 0.3 * want_frac
            if ai_user and s.startswith("aimodel_"):
                score += 0.02
            if p_val < 0.05:
                score += 0.05
            cand[s] = float(score)

        ranked = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

        recs = []
        for skill, score in ranked:
            roi = self.roi_map.get(skill, {})
            n_users = int(roi.get("users", 0))
            nbr_pct = float(neighbours[skill].mean() * 100)
            want_pct = float(want_counts[skill] / n_nbr * 100)
            recs.append({
                "skill": skill,
                "score": round(score, 4),
                "salary_impact_pct": round(float(roi.get("pct_salary_impact", float("nan"))), 1),
                "salary_p_value": round(float(roi.get("p_value", float("nan"))), 4),
                "salary_ci_pct": [
                    round(float(roi.get("ci_lo", float("nan"))), 1),
                    round(float(roi.get("ci_hi", float("nan"))), 1),
                ],
                "satisfaction_coef": (round(self.sat_map[skill], 3)
                            if skill in self.sat_map else None),
                "n_known": n_users,
                "nbr_prevalence_pct": round(nbr_pct, 1),
                "nbr_want_pct": round(want_pct, 1),
                "evidence": (
                    f"{nbr_pct:.0f}% of your {len(neighbours)} nearest peers know {skill} "
                    f"and {want_pct:.0f}% say they want to learn it; associated with "
                    f"{roi.get('pct_salary_impact', 0):+.1f}% log-comp change "
                    f"(p={roi.get('p_value', 1):.3g}, n={n_users:,})."
                ),
            })

        return {
            "persona_id": persona_id,
            "persona_top_skills": [s["skill"] for s in persona.get("top_skills", [])[:5]],
            "persona_size": int(persona.get("n", 0)),
            "n_neighbours": int(len(neighbours)),
            "salary_premium_log": float(salary_premium),
            "recommendations": recs,
        }

    # --- cold start ------------------------------------------------------------
    def _cold_start(self, top_n: int, df: pd.DataFrame | None = None) -> dict:
        """No skills entered: recommend the globally most lucrative skills
        within the optional country/experience filter (df may be the
        pre-filtered subset)."""
        if df is None:
            df = self._data()
        overall = df["log_comp"].mean()
        persona_id = max(self.personas, key=lambda c: self.personas[c]["n"])
        persona = self.personas[persona_id]
        cand = []
        for _, roi in self.roi.iterrows():
            s = roi["skill"]
            global_freq = float(df[s].mean())
            if global_freq < 0.03 or roi["coef_log_comp"] <= 0:
                continue
            score = float(roi["coef_log_comp"]) + 0.3 * global_freq
            if ai_user and str(roi["skill"]).startswith("aimodel_"):
                score += 0.02
            if roi["p_value"] < 0.05:
                score += 0.05
            cand.append((s, score, roi, global_freq))
        cand.sort(key=lambda t: t[1], reverse=True)
        recs = []
        for s, score, roi, freq in cand[:top_n]:
            recs.append({
                "skill": s,
                "score": round(score, 4),
                "salary_impact_pct": round(float(roi["pct_salary_impact"]), 1),
                "salary_p_value": round(float(roi["p_value"]), 4),
                "salary_ci_pct": [round(float(roi["ci_lo"]), 1),
                                   round(float(roi["ci_hi"]), 1)],
                "satisfaction_coef": (round(self.sat_map[s], 3)
                            if s in self.sat_map else None),
                "n_known": int(roi["users"]),
                "nbr_prevalence_pct": round(freq * 100, 1),
                "nbr_want_pct": round(freq * 100, 1),
                "evidence": (
                    f"Cold-start recommendation: {freq*100:.0f}% of the survey "
                    f"know {s}; associated with {roi['pct_salary_impact']:+.1f}% "
                    f"log-comp change (p={roi['p_value']:.3g}, n={int(roi['users']):,})."
                ),
            })
        return {
            "persona_id": int(persona_id),
            "persona_top_skills": [s["skill"] for s in persona.get("top_skills", [])[:5]],
            "persona_size": int(persona.get("n", 0)),
            "n_neighbours": 0,
            "salary_premium_log": 0.0,
            "recommendations": recs,
        }


if __name__ == "__main__":
    sr = SkillRadar()
    out = sr.recommend(current_skills=["lang_Python", "lang_JavaScript"], top_n=5)
    print(json.dumps(out, indent=2))
