# Media Metadata Intelligence System

Multi-label tagging, search, and recommendation for movie metadata.

This project builds a reproducible NLP system that predicts multiple content and story tags from a movie title and synopsis, produces structured metadata JSON, evaluates multi-label performance, and later supports semantic search and similar-content recommendation.

## Current Phase

Phase 1 focuses on the data foundation:

- Project scaffold and configuration
- MPST dataset loading and validation
- Multi-label tag parsing and binarisation utilities
- Conservative text preprocessing
- Exploratory data analysis reports and figures
- Processed dataset export for later modelling

No models are trained in this phase.

## Planned Phases

1. Dataset loading, validation, EDA, and preprocessing
2. Baseline multi-label classifiers and evaluation
3. Transformer or embedding-based tagging models
4. Structured metadata JSON generation
5. Semantic search and similar-content recommendation
6. FastAPI service and Streamlit interface

## Dataset

This project uses the MPST Movie Plot Synopses with Tags dataset.

Place the CSV file here:

```text
data/raw/mpst_full_data.csv
```

Expected columns:

- `imdb_id`
- `title`
- `plot_synopsis`
- `tags`
- `split`
- `synopsis_source`

The `tags` column should contain comma-separated multi-label tags.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run EDA

```bash
python scripts/run_eda.py
```

Outputs:

- `reports/eda_summary.json`
- `reports/top_tags.csv`
- figures in `reports/figures/`

## Prepare Data

```bash
python scripts/prepare_data.py
```

Outputs:

- `data/processed/mpst_processed.parquet`
- `data/processed/tag_classes.json`

## Tests

```bash
pytest
```
