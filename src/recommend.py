"""KNN-based upskilling recommender.

  1. Build the user's binary skill vector in the 86-skill cluster space.
  2. Find the k nearest developers in the survey by Hamming distance on
     the binary skill matrix (KNN -- Module 4).
  3. For every skill the user does NOT have, compute
        score(s) = neighbour_prevalence(s)
                 + 0.5 * regression_skill_ROI(s)
                 + 0.3 * neighbour_salary_premium
     and rank.
  4. Return the top N with the per-skill evidence strings.
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
                  k_neighbors: int = 300) -> dict:
        df = self._data()
        have = {s for s in current_skills}
        persona_id = self.persona_for(current_skills)
        persona = self.personas.get(persona_id, {})

        # Cold-start: no skills listed -> recommend globally high-ROI,
        # high-prevalence skills. Still evidence-backed, just not personalised.
        if not have:
            return self._cold_start(top_n)

        skills = self.skills
        v = np.array([1.0 if s in have else 0.0 for s in skills], dtype=float).reshape(1, -1)
        X = self._skill_matrix()
        if self._nn is None:
            self._nn = NearestNeighbors(
                n_neighbors=min(k_neighbors, X.shape[0]),
                metric="jaccard", n_jobs=-1,
            ).fit(X)
        dists, idxs = self._nn.kneighbors(v)
        # Drop neighbours that are identical-empty to the user (Jaccard undefined
        # = 0/0; sklearn returns dist=1 for them). Keep only those with finite,
        # non-trivial distance.
        keep = np.isfinite(dists[0])
        idxs = idxs[0][keep]
        if len(idxs) == 0:
            return self._cold_start(top_n)
        neighbours = df.iloc[idxs]
        nbr_log_comp = neighbours["log_comp"].mean()
        overall_log_comp = df["log_comp"].mean()
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
                "satisfaction_coef": round(float(self.sat_map.get(skill, 0.0)), 3),
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
    def _cold_start(self, top_n: int) -> dict:
        """No skills entered: recommend the globally most lucrative skills."""
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
                "satisfaction_coef": round(float(self.sat_map.get(s, 0.0)), 3),
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
