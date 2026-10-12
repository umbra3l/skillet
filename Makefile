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

data:
	bash data/download.sh

app:
	$(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -f data/processed.parquet data/processed_meta.json
	rm -f data/skill_roi.csv data/regression_summary.json
	rm -f data/satisfaction_coefs.csv data/satisfaction_metrics.json data/satisfaction_model.joblib
	rm -f data/cluster_model.joblib data/cluster_meta.json data/stats_tests.json
	rm -f assets/*.png
