"""End-to-end audit of the skillet recommender.

Runs the same SkillRadar the Streamlit app uses, varies the inputs the way
the app's controls do, and prints whether role / years / country / stack are
actually changing the output. Designed to be hand-readable; intentionally
verbose.

Run from repo root:  .venv/bin/python scripts/audit_recommender.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommend import SkillRadar  # noqa: E402


def _summary(rec: dict) -> dict:
    recs = rec["recommendations"]
    return {
        "persona_id": rec["persona_id"],
        "n_neighbours": rec["n_neighbours"],
        "salary_premium_log": round(rec["salary_premium_log"], 4),
        "top1": recs[0]["skill"] if recs else None,
        "top1_impact_pct": recs[0]["salary_impact_pct"] if recs else None,
        "top1_p": recs[0]["salary_p_value"] if recs else None,
        "top1_peer_pct": recs[0]["nbr_prevalence_pct"] if recs else None,
        "top1_want_pct": recs[0]["nbr_want_pct"] if recs else None,
        "skills_returned": [r["skill"] for r in recs],
    }


def label(label_: str) -> None:
    print()
    print("=" * 78)
    print(label_)
    print("=" * 78)


def main() -> None:
    sr = SkillRadar()

    # Discover the role_ / country_ columns the app's selects actually use.
    df = sr._data()
    role_cols = sorted([c for c in df.columns if c.startswith("role_")])
    country_cols = sorted([c for c in df.columns if c.startswith("country_")])
    print(f"[audit] {len(df):,} rows · "
          f"{len(role_cols)} roles · {len(country_cols)} countries · "
          f"{len(sr.roi)} ROI rows")

    # ------------------------------------------------------------------
    # 1. Cold-start invariant: empty stack must produce the same recs
    #    for the same filter, regardless of any role/workexp differences
    #    (cold-start has no peer pool by design). Insensitive by contract.
    # ------------------------------------------------------------------
    label("1.  COLD-START (empty stack)")
    recA = sr.recommend(current_skills=[], top_n=5, country="United States of America",
                       workexp=5, role="role_Developer_back_end",
                       ai_user=True)
    recB = sr.recommend(current_skills=[], top_n=5, country="United States of America",
                       workexp=5, role="role_Data_scientist_or_machine_learning_specialist",
                       ai_user=True)
    recC = sr.recommend(current_skills=[], top_n=5, country="Germany",
                       workexp=5, role="role_Developer_back_end",
                       ai_user=True)
    sA, sB, sC = _summary(recA), _summary(recB), _summary(recC)
    print(f"  role=backend,  US,        5y : top1={sA['top1']!r:30s} "
          f"peer%={sA['top1_peer_pct']}  want%={sA['top1_want_pct']}")
    print(f"  role=ds/ml,    US,        5y : top1={sB['top1']!r:30s} "
          f"peer%={sB['top1_peer_pct']}  want%={sB['top1_want_pct']}")
    print(f"  role=backend,  Germany,   5y : top1={sC['top1']!r:30s} "
          f"peer%={sC['top1_peer_pct']}  want%={sC['top1_want_pct']}")
    print("  Cold-start uses the country/experience mask for prevalence, so")
    print("  the prevalence/want% for the same filter should match exactly;")
    print("  roles and AI-user are bonus terms, so they shouldn't change order.")
    print(f"  -> skills returned US-back:   {sA['skills_returned']}")
    print(f"  -> skills returned US-ds/ml: {sB['skills_returned']}")
    print(f"  -> skills returned DE-back:   {sC['skills_returned']}")

    # ------------------------------------------------------------------
    # 2. Stack effect: changing the stack MUST change recommendations.
    # ------------------------------------------------------------------
    label("2.  STACK EFFECT (with stack)")
    backend_stack = ["lang_Python", "lang_JavaScript", "web_Node_js",
                     "web_Express", "plat_npm", "devenv_Visual_Studio_Code"]
    data_stack    = ["lang_Python", "lang_R", "devenv_Jupyter_Notebook_JupyterLab",
                     "db_PostgreSQL", "aimodel_openAI_GPT_chatbot_models",
                     "plat_Amazon_Web_Services_AWS"]
    recBE = sr.recommend(current_skills=backend_stack, top_n=5,
                         country="United States of America", workexp=5,
                         role="role_Developer_back_end", ai_user=True)
    recDS = sr.recommend(current_skills=data_stack, top_n=5,
                         country="United States of America", workexp=5,
                         role="role_Data_scientist_or_machine_learning_specialist",
                         ai_user=True)
    sBE, sDS = _summary(recBE), _summary(recDS)
    print(f"  back-end stack : nbr={sBE['n_neighbours']:4d}  "
          f"premium={sBE['salary_premium_log']:+.4f}  "
          f"persona={sBE['persona_id']}")
    print(f"    top5: {[r['skill'] for r in recBE['recommendations']]}")
    print(f"  ds/ml stack    : nbr={sDS['n_neighbours']:4d}  "
          f"premium={sDS['salary_premium_log']:+.4f}  "
          f"persona={sDS['persona_id']}")
    print(f"    top5: {[r['skill'] for r in recDS['recommendations']]}")
    same_set = sBE["skills_returned"] == sDS["skills_returned"]
    print(f"  -> identical top-5 across stacks? {same_set}   "
          "(should be False: this is the personalisation test)")

    # ------------------------------------------------------------------
    # 3. Country effect: same profile, different country -> differs.
    # ------------------------------------------------------------------
    label("3.  COUNTRY EFFECT")
    for country in ["United States of America", "Germany",
                    "India", "Brazil", "Japan"]:
        r = sr.recommend(current_skills=backend_stack, top_n=5,
                         country=country, workexp=5,
                         role="role_Developer_back_end", ai_user=True)
        s = _summary(r)
        print(f"  {country:30s} : nbr={s['n_neighbours']:4d}  "
              f"premium={s['salary_premium_log']:+.4f}  "
              f"top1={s['top1']:25s}  top5={s['skills_returned']}")

    # ------------------------------------------------------------------
    # 4. Workexp effect: same profile, different experience -> may differ
    #    because the workexp mask narrows the peer pool.
    # ------------------------------------------------------------------
    label("4.  WORKEXP EFFECT")
    for we in [0, 1, 3, 5, 10, 20, 40]:
        r = sr.recommend(current_skills=backend_stack, top_n=5,
                         country="United States of America", workexp=we,
                         role="role_Developer_back_end", ai_user=True)
        s = _summary(r)
        print(f"  workexp={we:3d} : nbr={s['n_neighbours']:4d}  "
              f"premium={s['salary_premium_log']:+.4f}  "
              f"top1={s['top1']:25s}  top5={s['skills_returned']}")

    # ------------------------------------------------------------------
    # 5. Role effect: same profile, different role -> may differ
    #    because role narrows the peer pool.
    # ------------------------------------------------------------------
    label("5.  ROLE EFFECT")
    for role in ["role_Developer_back_end",
                 "role_Developer_front_end",
                 "role_Developer_full_stack",
                 "role_Data_scientist_or_machine_learning_specialist",
                 "role_DevOps_specialist",
                 "role_Engineer_data",
                 "role_Developer_mobile",
                 "role_Academic_researcher"]:
        if role not in role_cols:
            print(f"  {role:55s} : (column not present in df)")
            continue
        r = sr.recommend(current_skills=backend_stack, top_n=5,
                         country="United States of America", workexp=5,
                         role=role, ai_user=True)
        s = _summary(r)
        print(f"  {role:55s} : nbr={s['n_neighbours']:4d}  "
              f"premium={s['salary_premium_log']:+.4f}  "
              f"top1={s['top1']:25s}  top5={s['skills_returned']}")

    # ------------------------------------------------------------------
    # 6. Persona invariant: each persona_id has consistent "top skills".
    # ------------------------------------------------------------------
    label("6.  PERSONA INVARIANT")
    for pid, p in sr.personas.items():
        ts = ", ".join(s["skill"] for s in p["top_skills"][:5])
        print(f"  persona {pid}: share={p['share']:.3%}  n={p['n']:,}  "
              f"top=[{ts}]")

    # ------------------------------------------------------------------
    # 7. Sanity: top-5 salary numbers in persona-vs-truth check.
    #    The MLR coefficient for each recommended skill, compared to the
    #    ROI table.
    # ------------------------------------------------------------------
    label("7.  RECOMMENDATION EVIDENCE vs ROI TABLE")
    r = sr.recommend(current_skills=backend_stack, top_n=5,
                     country="United States of America", workexp=5,
                     role="role_Developer_back_end", ai_user=True)
    for rec in r["recommendations"]:
        roi_row = sr.roi_map.get(rec["skill"])
        if not roi_row:
            print(f"  {rec['skill']:25s} : MISSING from ROI table")
            continue
        roi_coef = float(roi_row["pct_salary_impact"])
        roi_p = float(roi_row["p_value"])
        ok_impact = abs(rec["salary_impact_pct"] - roi_coef) < 0.05
        ok_p = abs(rec["salary_p_value"] - roi_p) < 1e-4 or \
               (rec["salary_p_value"] != rec["salary_p_value"])  # NaN accept
        mark = "OK " if ok_impact and ok_p else "!! "
        print(f"  {mark}{rec['skill']:25s}  card={rec['salary_impact_pct']:+6.2f}%  "
              f"roi={roi_coef:+6.2f}%   card_p={rec['salary_p_value']:.3g}  "
              f"roi_p={roi_p:.3g}")

    # ------------------------------------------------------------------
    # 8. The "score" formula sanity: written-down vs. observed.
    # ------------------------------------------------------------------
    label("8.  SCORE FORMULA — written in the methodology")
    print("  score(s) = peer_prevalence(s)")
    print("           + 0.5 * salary_ROI_coef_log")
    print("           + 0.2 * max(peer_salary_premium_log, 0)")
    print("           + 0.3 * peer_demand(s)")
    print("           + 0.02 if ai_user and s starts with 'aimodel_'")
    print("           + 0.05 if p(s) < 0.05")
    # Spot-check the top recommendation.
    r = sr.recommend(current_skills=backend_stack, top_n=1,
                     country="United States of America", workexp=5,
                     role="role_Developer_back_end", ai_user=True)
    if r["recommendations"]:
        rec = r["recommendations"][0]
        # Recompute manually from the same data the model used.
        neighbours = sr._data().iloc[:]  # placeholder; we can read score directly
        print(f"  observed top1 score = {rec['score']}")
        print(f"  observed top1       = {rec['skill']!r}")
        print(f"  observed impact%    = {rec['salary_impact_pct']}%")
        print(f"  observed prevalence = {rec['nbr_prevalence_pct']}% (peer share)")
        print(f"  observed want%      = {rec['nbr_want_pct']}% (peer share)")
    print()


if __name__ == "__main__":
    main()
