"""KNN-based upskilling recommender.

  1. Build the user's binary skill vector over every skill column the app
     offers (122 skills).
  2. Find the k nearest developers in the survey by Jaccard distance on
     the binary skill matrix (KNN -- Module 4). The KNN is built once on
     the full dataset and ranks every developer; the recommender keeps
     those who match the visitor's role, country, experience (+/- 10
     years) and AI-tool use. When fewer than MIN_PEERS developers match,
     the filters are relaxed one at a time in RELAX_ORDER -- AI-tool use,
     then experience, then country, then role -- and the result says
     which filters were applied. Each of the k peers is weighted by its
     Jaccard similarity to the user's stack (1 - distance), so the closest
     peers count most even when the filtered pool is smaller than k.
  3. For every skill the user does NOT have, compute
        score(s) =  peer_prevalence(s)                   # similarity-weighted share of your filtered peers
                + 0.5 * salary_ROI(s)                   # per-skill MLR coefficient (log pts)
                + 0.3 * peer_demand(s)                  # share of peers who "want" to learn it
                + 0.02 if ai_user and s starts with "aimodel_"   # small nudge
                + 0.05 if p(s) < 0.05                   # significance bonus
     and rank. The salary-ROI coefficient is the partial effect from
     log_comp ~ work_exp + country + skill, so it is already conditioned
     on work experience and country; the sidebar inputs additionally
     shape *which peers* the score uses.
     The peer salary premium is reported alongside but is not part of the
     score: it is the same for every candidate skill, so it cannot change
     the ranking. It is the peers' mean residual from
     log_comp ~ work_exp + country, i.e. how much more the peers earn
     than developers with the same experience in the same country.
  4. Cold-start (no skills listed): the same salary-ROI prior, with the
     global skill prevalence (within the same profile filter) and the
     +0.05 significance bonus. No peer_prevalence, peer_salary premium,
     or peer_demand terms exist in cold start -- there is no peer group
     yet.
  5. The persona is the saved K-means cluster (src/cluster.py) of the
     user's stack, restricted to the 86 cluster-space skills.
  6. Return the top N with the per-skill evidence strings. The
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
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Smallest peer group the recommender will personalise from. Below this,
# the profile filters are relaxed in RELAX_ORDER until the pool is big enough.
MIN_PEERS = 50
RELAX_ORDER = ("ai", "experience", "country", "role")
# Score weights (see the module docstring). src/evaluate.py validates them
# against held-out respondents.
WEIGHTS = {"roi": 0.5, "want": 0.3, "sig": 0.05, "ai": 0.02}
SKILL_PREFIXES = ("lang_", "db_", "plat_", "web_", "devenv_", "aimodel_")


class SkillRadar:
    def __init__(self) -> None:
        self._cluster = joblib.load(DATA / "cluster_model.joblib")
        self.cluster_skills: list[str] = self._cluster["skills"]  # the 86 persona-space skills
        columns = pq.read_schema(DATA / "processed.parquet").names
        # Every skill the app offers: the KNN and the candidate list use all of them.
        self.skills: list[str] = [c for c in columns if c.startswith(SKILL_PREFIXES)]
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
        self._resid: np.ndarray | None = None

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

    def _pay_residuals(self) -> np.ndarray:
        """Residual of log_comp ~ WorkExp + country fixed effects, per row.

        NaN for rows outside the salary model sample (no USD pay, or an IQR
        outlier). The mean residual of a peer group is its pay premium over
        developers with the same experience in the same country.
        """
        if self._resid is None:
            from src.regression import country_effects
            df = self._data()
            ok = ((df["comp_inlier"] == 1) & df["log_comp"].notna()).to_numpy()
            sub = df[ok]
            X = np.column_stack([np.ones(len(sub)), sub["WorkExp"].to_numpy(float),
                                 country_effects(sub).to_numpy()])
            y = sub["log_comp"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = np.full(len(df), np.nan)
            resid[ok] = y - X @ beta
            self._resid = resid
        return self._resid

    # --- persona (the fitted K-means cluster) ----------------------------------
    def persona_for(self, current_skills: Iterable[str]) -> int:
        """Assign the stack to a persona with the saved scaler -> PCA -> K-means.

        A stack with none of the 86 cluster-space skills carries no persona
        signal, so it gets the largest persona.
        """
        have = set(current_skills)
        if not have & set(self.cluster_skills):
            return int(max(self.personas, key=lambda c: self.personas[c]["n"]))
        v = np.array([[1.0 if s in have else 0.0 for s in self.cluster_skills]])
        z = self._cluster["pca"].transform(self._cluster["scaler"].transform(v))
        return int(self._cluster["kmeans"].predict(z)[0])

    # --- KNN recommender -------------------------------------------------------
    def recommend(self, current_skills: Iterable[str], top_n: int = 5,
                  k_neighbors: int = 300,
                  country: str | None = None,
                  workexp: int | None = None,
                  role: str | None = None,
                  ai_user: bool | None = None,
                  weights: dict[str, float] | None = None,
                  exclude_row: int | None = None) -> dict:
        """Top-N skills to learn next.

        `weights` overrides entries of WEIGHTS. `exclude_row` removes one
        survey row from the peer pool (used by src/evaluate.py so a held-out
        respondent is not their own peer).
        """
        wt = {**WEIGHTS, **(weights or {})}
        df_full = self._data()
        have = {s for s in current_skills}
        persona_id = self.persona_for(current_skills)
        persona = self.personas.get(persona_id, {})

        mask, peer_filter = self._peer_mask(country, workexp, role, ai_user)
        if exclude_row is not None:
            mask[exclude_row] = False

        # Cold-start: no skills listed -> recommend globally high-ROI,
        # high-prevalence skills within the (possibly relaxed) filter.
        if not have:
            out = self._cold_start(top_n, df_full[mask], ai_user=ai_user)
            out["peer_filter"] = peer_filter
            out.update(self._group_stats(np.flatnonzero(mask)))
            return out

        skills = self.skills
        v = np.array([1.0 if s in have else 0.0 for s in skills], dtype=float).reshape(1, -1)
        X = self._skill_matrix()
        if self._nn is None:
            self._nn = NearestNeighbors(
                n_neighbors=X.shape[0], metric="jaccard", n_jobs=-1,
            ).fit(X)
        # Rank every developer by distance, then keep only those that pass
        # the profile mask. Ranking all rows (not a top-5000 shortlist)
        # means every matching developer is reachable as a peer.
        dists, idxs = self._nn.kneighbors(v, n_neighbors=self._nn.n_neighbors)
        keep = np.isfinite(dists[0])
        cand, dist = idxs[0][keep], dists[0][keep]
        in_pool = mask[cand]
        cand, dist = cand[in_pool], dist[in_pool]
        if cand.size == 0:
            out = self._cold_start(top_n, df_full[mask], ai_user=ai_user)
            out["peer_filter"] = peer_filter
            out.update(self._group_stats(np.flatnonzero(mask)))
            return out
        # Take the top k_neighbors from the filtered set and weight each by
        # its Jaccard similarity (1 - distance). A filtered pool is often
        # smaller than k, so without weights every pool member would count
        # equally and the stack would not shape the peer group at all.
        cand, dist = cand[:k_neighbors], dist[:k_neighbors]
        w = 1.0 - dist
        if w.sum() <= 0:  # nobody shares a single skill: fall back to equal weights
            w = np.ones_like(w)
        w = w / w.sum()
        neighbours = df_full.iloc[cand]
        group_median_usd = self._group_stats(cand)["group_median_usd"]
        nbr_resid = self._pay_residuals()[cand]
        has_pay = np.isfinite(nbr_resid)
        salary_premium = (float(np.average(nbr_resid[has_pay], weights=w[has_pay]))
                          if has_pay.any() and w[has_pay].sum() > 0 else 0.0)

        # WantToWorkWith demand signal (lazy-loaded; first call reads the raw CSV)
        if self._want is None:
            from src.want import load_want_sets
            self._want = load_want_sets()
        # Similarity-weighted share of peers who want / know each skill.
        want_share: dict[str, float] = {s: 0.0 for s in skills}
        for nid, wi in zip(neighbours["id"].astype(str), w):
            for s in self._want.get(nid, ()):
                if s in want_share:
                    want_share[s] += wi
        know_share = dict(zip(skills, w @ neighbours[skills].to_numpy(float)))

        cand: dict[str, float] = {}
        for s in skills:
            if s in have:
                continue
            freq = float(know_share[s])
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
            want_frac = want_share[s]
            score = freq + wt["roi"] * roi_coef + wt["want"] * want_frac
            if ai_user and s.startswith("aimodel_"):
                score += wt["ai"]
            if p_val < 0.05:
                score += wt["sig"]
            cand[s] = float(score)

        ranked = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

        recs = []
        for skill, score in ranked:
            roi = self.roi_map.get(skill, {})
            n_users = int(roi.get("users", 0))
            nbr_pct = float(know_share[skill] * 100)
            want_pct = float(want_share[skill] * 100)
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
                    f"Weighted by similarity, {nbr_pct:.0f}% of your {len(neighbours)} "
                    f"nearest peers know {skill} "
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
            "group_median_usd": group_median_usd,
            "peer_filter": peer_filter,
            "recommendations": recs,
        }

    def _group_stats(self, rows: np.ndarray) -> dict:
        """Pay context for a set of survey rows: the median USD pay, and the
        mean residual from log_comp ~ WorkExp + country (their premium over
        developers with the same experience in the same country)."""
        df = self._data()
        pay = df["CompUSD"].to_numpy()[rows]
        resid = self._pay_residuals()[rows]
        ok = np.isfinite(resid)
        return {
            "group_median_usd": float(np.nanmedian(pay)) if np.isfinite(pay).any() else float("nan"),
            "salary_premium_log": float(resid[ok].mean()) if ok.any() else 0.0,
        }

    # --- peer filter -----------------------------------------------------------
    def _peer_mask(self, country: str | None, workexp: int | None,
                   role: str | None, ai_user: bool | None = None,
                   ) -> tuple[np.ndarray, dict]:
        """Rows matching the visitor's profile, relaxed one filter at a time.

        Each requested filter (role, country, experience +/- 10 years,
        AI-tool use) is applied; while fewer than MIN_PEERS developers
        match, the next
        filter in RELAX_ORDER is dropped. Returns the mask and a summary:
        {"requested", "applied", "dropped", "n_pool"}.
        """
        df = self._data()
        conds: dict[str, np.ndarray] = {}
        if role is not None and role in df.columns:
            # `role` is the raw column name (e.g. "role_Developer_back_end").
            conds["role"] = df[role].values == 1
        if country is not None:
            conds["country"] = df["Country"].values == country
        if workexp is not None:
            lo, hi = max(0, int(workexp) - 10), int(workexp) + 10
            conds["experience"] = (df["WorkExp"].values >= lo) & (df["WorkExp"].values <= hi)
        if ai_user is not None:
            conds["ai"] = df["ai_user"].values == int(bool(ai_user))

        requested = [f for f in ("role", "country", "experience", "ai") if f in conds]
        applied = list(requested)
        while True:
            mask = np.ones(len(df), dtype=bool)
            for f in applied:
                mask &= conds[f]
            if mask.sum() >= MIN_PEERS or not applied:
                break
            applied.remove(next(f for f in RELAX_ORDER if f in applied))
        return mask, {
            "requested": requested,
            "applied": applied,
            "dropped": [f for f in requested if f not in applied],
            "n_pool": int(mask.sum()),
        }

    # --- cold start ------------------------------------------------------------
    def _cold_start(self, top_n: int, df: pd.DataFrame | None = None,
                 ai_user: bool | None = None) -> dict:
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
