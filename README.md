# Media Metadata Intelligence System

Multi-label tagging, search, and recommendation for movie metadata.

This project builds a reproducible NLP system that predicts multiple content and story tags from a movie title and synopsis, produces structured metadata JSON, evaluates multi-label performance, and later supports semantic search and similar-content recommendation.

## Current Phase

Phase 1 focused on the data foundation:

- Project scaffold and configuration
- MPST dataset loading and validation
- Multi-label tag parsing and binarisation utilities
- Conservative text preprocessing
- Exploratory data analysis reports and figures
- Processed dataset export for later modelling

Phase 2 adds a classical machine learning baseline:

- TF-IDF text features from combined title and synopsis
- OneVsRest logistic regression for multi-label tag prediction
- Validation-only threshold tuning
- Final test-set evaluation
- Structured JSON prediction output

Phase 3 adds semantic retrieval:

- Sentence embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Cosine nearest-neighbor search with scikit-learn
- Reusable local search artifacts
- Similar-content and combined intelligence report CLIs

## Planned Phases

1. Dataset loading, validation, EDA, and preprocessing
2. TF-IDF + OneVsRest logistic regression baseline and evaluation
3. Semantic search and similar-content recommendation
4. Structured metadata JSON generation
5. Transformer or embedding-based tagging models
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

## Train Baseline

```bash
python scripts/train_baseline.py
```

The baseline uses the original MPST split column:

- `train` rows are used only for fitting the TF-IDF + logistic regression model.
- `val` rows are used only to tune one global probability threshold.
- `test` rows are used only for final evaluation.

Outputs:

- `models/baseline_tfidf_logreg.joblib`
- `reports/baseline_metrics.json`
- `reports/baseline_thresholds.csv`
- `reports/baseline_per_tag_report.csv`

Trained model files are ignored by git.

## Predict Tags

```bash
python scripts/predict_tags.py \
  --title "Example Film" \
  --synopsis "A detective investigates a violent murder while uncovering a dark family secret." \
  --threshold 0.35
```

The prediction script prints structured JSON with predicted tags sorted by confidence. If `--threshold` is omitted, it uses the tuned threshold from `reports/baseline_metrics.json` when available.

## Build Search Index

```bash
python scripts/build_search_index.py
```

This embeds `combined_text` from `data/processed/mpst_processed.parquet` and saves reusable search artifacts:

- `artifacts/search/mpst_embeddings.npy`
- `artifacts/search/mpst_search_metadata.parquet`
- `artifacts/search/mpst_nn_index.joblib`

Generated search artifacts are ignored by git.

## Search Similar Content

```bash
python scripts/search_similar.py \
  --title "Example Film" \
  --synopsis "A detective investigates a violent murder while uncovering a dark family secret." \
  --top-k 5
```

The script returns structured JSON containing ranked similar MPST movies, cosine similarity scores, tags, metadata, and short synopsis previews. Results are deduplicated by `imdb_id` and normalised title so repeated dataset entries do not crowd out the recommendations.

## Intelligence Report

```bash
python scripts/intelligence_report.py \
  --title "Example Film" \
  --synopsis "A detective investigates a violent murder while uncovering a dark family secret." \
  --top-k 5
```

This combines Phase 2 tag prediction with Phase 3 semantic search in one JSON object. It requires both the trained baseline model and the search artifacts.

## Retrieval Concepts

- Tag prediction assigns likely MPST labels to a new title and synopsis.
- Semantic search embeds text and retrieves movies with similar meaning, even when exact words differ.
- Similar-content recommendation uses semantic search results as candidates for "more like this" experiences.

## Baseline Metrics

- `micro-F1` measures global precision/recall balance across all movie-tag decisions. Frequent tags have more influence.
- `macro-F1` averages F1 across tags. Rare tags matter as much as common tags, so this is usually harder.
- `Hamming loss` is the fraction of individual tag decisions that are wrong. Lower is better.

Micro and macro precision/recall are also reported to show whether the model is over-predicting or under-predicting tags.

FastAPI and Streamlit are planned for later phases.

## Tests

```bash
pytest
```
