.PHONY: data app all clean

PY ?= .venv/bin/python

all: data
	$(PY) run_all.py
	$(PY) notebooks/01_eda.py
	$(PY) notebooks/02_stats_tests.py
	$(PY) notebooks/03_regression.py
	$(PY) notebooks/04_classification.py
	$(PY) notebooks/05_clustering.py
	$(PY) notebooks/06_recommend.py
	$(PY) -m src.fa
	$(PY) -m src.anova
	$(PY) -m src.network
	$(PY) notebooks/07_fa_anova.py
	$(PY) notebooks/08_network.py

# Multi-year trend (downloads 2022-2025 first, then fits per-year MLRs).
# This is slow because of the four ~100MB downloads; run it explicitly.
trend:
	$(PY) -m src.trend

data:
	bash data/download.sh

app:
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -f data/processed.parquet data/processed_meta.json
	rm -f data/skill_roi.csv data/regression_summary.json
	rm -f data/satisfaction_coefs.csv data/satisfaction_metrics.json data/satisfaction_model.joblib
	rm -f data/satisfaction_model_dt.joblib data/satisfaction_model_nb.joblib
	rm -f data/cluster_model.joblib data/cluster_meta.json data/stats_tests.json
	rm -f data/factor_loadings.csv data/fa_meta.json
	rm -f data/anova_personas.json
	rm -f data/skill_network.json data/network_meta.json
	rm -f data/trend_roi.csv data/trend_summary.json
	rm -f assets/*.png
