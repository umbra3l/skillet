"""Recommender engine: combine skill ROI + cluster persona + satisfaction coefs.

  Given a user profile (role, WorkExp, current skills, country, AI use),
  return the top-N next skills to learn, with the statistical evidence
  for each suggestion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class SkillRadar:
    def __init__(self):
        cluster = joblib.load(DATA / "cluster_model.joblib")
        self.skills: list[str] = cluster["skills"]
        self.pca = cluster["pca"]
        self.km = cluster["kmeans"]
        self.labels = cluster["labels"]
        self.Xp = cluster["Xp"]
        self.roi = pd.read_csv(DATA / "skill_roi.csv")
        self.roi_map: dict[str, dict] = {r["skill"]: r.to_dict() for _, r in self.roi.iterrows()}
        sat = pd.read_csv(DATA / "satisfaction_coefs.csv")
        self.sat_map: dict[str, float] = dict(zip(sat["feature"], sat["coef"]))
        meta = json.loads((DATA / "cluster_meta.json").read_text())
        self.personas = {p["cluster"]: p for p in meta["personas"]}
        self.best_k = meta["best_k"]
        # Full processed dataset for neighbor lookups (load on demand)
        self._df = None
        self._Xp_df = None

    def _load_df(self):
        if self._df is None:
            self._df = pd.read_parquet(DATA / "processed.parquet")
            self._Xp_df = self._Xp_df  # placeholder
        return self._df

    def _skill_vector(self, current_skills: Iterable[str]) -> np.ndarray:
        v = np.zeros(len(self.skills), dtype=float)
        for s in current_skills:
            for i, name in enumerate(self.skills):
                if s.lower() == name.lower() or s.lower() == name[len(name.split('_', 1)[0] + '_'):]:
                    v[i] = 1
                    break
        return v

    def persona_for(self, current_skills: Iterable[str]) -> int:
        v = self._skill_vector(current_skills).reshape(1, -1)
        Xs = (v - v.mean()) / (v.std() + 1e-9)
        # Match the scaler used at training time -- we need it from disk
        sc = joblib.load(DATA / "cluster_model.joblib")["scaler"]
        Xp = self.pca.transform(sc.transform(v))
        return int(self.km.predict(Xp)[0])

    def recommend(self, current_skills: Iterable[str], top_n: int = 5) -> dict:
        skills_have = list(current_skills)
        v = self._skill_vector(skills_have)
        persona_id = self.persona_for(skills_have)
        persona = self.personas.get(persona_id, {})

        # Candidate skills: present in this persona's top skills OR globally strong,
        # but the user does NOT already have.
        have_set = {s.lower() for s in skills_have}
        candidates: dict[str, float] = {}
        for s in self.skills:
            if s.lower() in have_set:
                continue
            roi = self.roi_map.get(s)
            if not roi:
                continue
            # Score = standardized ROI coef + satisfaction coef + persona prevalence
            sat = self.sat_map.get(s, 0.0)
            # Find this skill in persona top list (use lift if available)
            lift = 0.0
            for ts in persona.get("top_skills", []):
                if ts["skill"] == s:
                    lift = ts["lift"]
                    break
            score = (roi["coef_log_comp"] * 1.0) + (sat * 0.5) + (np.log1p(max(lift - 1, 0)) * 0.3)
            if roi["p_value"] < 0.05:
                score += 0.1  # small bonus for statistical significance
            candidates[s] = float(score)

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        recs = []
        for skill, score in ranked:
            roi = self.roi_map.get(skill, {})
            sat = self.sat_map.get(skill, 0.0)
            recs.append({
                "skill": skill,
                "score": round(score, 4),
                "salary_impact_pct": round(roi.get("pct_salary_impact", float("nan")), 1),
                "salary_p_value": round(roi.get("p_value", float("nan")), 4),
                "salary_ci_pct": [round(roi.get("ci_lo", float("nan")), 1),
                                   round(roi.get("ci_hi", float("nan")), 1)],
                "satisfaction_coef": round(float(sat), 3),
                "n_known": int(roi.get("users", 0)),
                "evidence": (
                    f"Knowing {skill} is associated with "
                    f"{roi.get('pct_salary_impact', 0):+.1f}% log-comp change "
                    f"(p={roi.get('p_value', 1):.3g}, n={int(roi.get('users', 0)):,})"
                    + (f"; satisfaction coef = {sat:+.3f}" if sat else "")
                ),
            })

        return {
            "persona_id": persona_id,
            "persona_top_skills": [s["skill"] for s in persona.get("top_skills", [])[:5]],
            "persona_size": int(persona.get("n", 0)),
            "recommendations": recs,
        }


if __name__ == "__main__":
    sr = SkillRadar()
    out = sr.recommend(current_skills=["lang_Python", "lang_JavaScript"], top_n=5)
    print(json.dumps(out, indent=2))
