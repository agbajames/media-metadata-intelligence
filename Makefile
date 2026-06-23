PYTHON ?= python

SAMPLE_TITLE := Example Film
SAMPLE_SYNOPSIS := A detective investigates a violent murder while uncovering a dark family secret.

.PHONY: install test prepare-data eda train-baseline build-search-index predict-sample search-sample intelligence-sample run-api run-streamlit

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

prepare-data:
	$(PYTHON) scripts/prepare_data.py

eda:
	$(PYTHON) scripts/run_eda.py

train-baseline:
	$(PYTHON) scripts/train_baseline.py

build-search-index:
	$(PYTHON) scripts/build_search_index.py

predict-sample:
	$(PYTHON) scripts/predict_tags.py --title "$(SAMPLE_TITLE)" --synopsis "$(SAMPLE_SYNOPSIS)" --threshold 0.35

search-sample:
	$(PYTHON) scripts/search_similar.py --title "$(SAMPLE_TITLE)" --synopsis "$(SAMPLE_SYNOPSIS)" --top-k 5

intelligence-sample:
	$(PYTHON) scripts/intelligence_report.py --title "$(SAMPLE_TITLE)" --synopsis "$(SAMPLE_SYNOPSIS)" --top-k 5

run-api:
	$(PYTHON) scripts/run_api.py

run-streamlit:
	$(PYTHON) scripts/run_streamlit.py
