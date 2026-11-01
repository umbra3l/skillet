"""End-to-end runner: preprocess -> stats -> regression -> classification -> clustering."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # run_all.py sits at the repo root
sys.path.insert(0, str(ROOT))


def step(name: str, fn) -> None:
    t0 = time.time()
    print(f"\n========== {name} ==========")
    fn()
    print(f"  -> {name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    from src import preprocess, stats_tests, regression, classification, cluster, want  # noqa: E402
    step("preprocess",     preprocess.build_processed)
    step("want extract",   want.build_extract)
    step("stats_tests",    stats_tests.run)
    step("regression",     regression.run)
    step("classification", classification.run)
    step("clustering",     cluster.run)
    print("\nAll artifacts written to data/. Now launch the app:")
    print("  streamlit run app/streamlit_app.py")
