# Media Metadata Intelligence System – Multi-label Tagging, Search and Recommendation

An end-to-end NLP portfolio project for turning movie titles and synopses into structured metadata, predicted story tags and similar-content recommendations.

## What It Does

This system:

- predicts multiple movie/story tags from a title and synopsis
- returns confidence scores for predicted tags
- produces structured metadata JSON for downstream products and APIs
- evaluates multi-label performance with micro-F1, macro-F1 and Hamming loss
- supports semantic search and similar-content recommendation
- exposes CLI, FastAPI and Streamlit interfaces

The classification model is a classical TF-IDF + OneVsRest Logistic Regression baseline. Sentence embeddings are used separately for semantic search and recommendation.

## Project Status

- [x] Data pipeline complete
- [x] Baseline multi-label model complete
- [x] Evaluation pipeline complete
- [x] Semantic search complete
- [x] FastAPI complete
- [x] Streamlit demo complete
- [x] CI added
- [ ] Docker/deployment planned
- [ ] Advanced transformer classifier planned

## Dataset

The project uses the MPST Movie Plot Synopses with Tags dataset.

Place the raw CSV locally at:

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

The raw dataset is not committed to this repository. Processed data, trained models, embeddings and search indexes are generated locally and ignored by Git.

## Architecture Overview

The system is split into five layers:

- Data pipeline – validates the MPST CSV, parses comma-separated labels, combines title and synopsis, and prepares local parquet data.
- Baseline tag model – uses TF-IDF text features with OneVsRest Logistic Regression for multi-label classification.
- Semantic search layer – embeds text with `sentence-transformers/all-MiniLM-L6-v2`, indexes embeddings with sklearn `NearestNeighbors`, and deduplicates recommendations by `imdb_id` and normalised title.
- API layer – exposes health, tag prediction, semantic search and combined intelligence report endpoints through FastAPI.
- Streamlit demo layer – provides an interview-ready UI for exploring predictions, recommendations and structured JSON output.

More detail is available in [docs/architecture.md](docs/architecture.md).

## Results

Current Phase 2 baseline performance on the MPST test split:

| Metric | Value |
| --- | ---: |
| selected threshold | 0.50 |
| micro_f1 | 0.4106 |
| macro_f1 | 0.1840 |
| hamming_loss | 0.0525 |
| micro_precision | 0.3958 |
| micro_recall | 0.4265 |
| macro_precision | 0.2561 |
| macro_recall | 0.1883 |

Macro-F1 is low because many MPST tags are rare and the dataset is imbalanced. The model is intended as a transparent baseline for metadata and recommendation workflows, not as perfect content truth.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional environment variables can be copied from `.env.example`.

## Reproduce The Pipeline

Run exploratory analysis:

```bash
python scripts/run_eda.py
```

Prepare processed data:

```bash
python scripts/prepare_data.py
```

Train the baseline classifier:

```bash
python scripts/train_baseline.py
```

Build the semantic search index:

```bash
python scripts/build_search_index.py
```

## CLI Examples

Predict tags:

```bash
python scripts/predict_tags.py \
  --title "Example Film" \
  --synopsis "A detective investigates a violent murder while uncovering a dark family secret." \
  --threshold 0.35
```

Search similar content:

```bash
python scripts/search_similar.py \
  --title "Example Film" \
  --synopsis "A detective investigates a violent murder while uncovering a dark family secret." \
  --top-k 5
```

Generate a combined intelligence report:

```bash
python scripts/intelligence_report.py \
  --title "Example Film" \
  --synopsis "A detective investigates a violent murder while uncovering a dark family secret." \
  --top-k 5
```

Sample outputs are documented in [docs/sample_outputs.md](docs/sample_outputs.md).

## FastAPI

Ensure local artifacts exist first:

```bash
python scripts/train_baseline.py
python scripts/build_search_index.py
```

Run the API:

```bash
python scripts/run_api.py
```

Default URL:

```text
http://127.0.0.1:8000
```

Endpoints:

- `GET /`
- `GET /health`
- `POST /predict-tags`
- `POST /search-similar`
- `POST /intelligence-report`

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Predict tags:

```bash
curl -X POST http://127.0.0.1:8000/predict-tags \
  -H "Content-Type: application/json" \
  -d '{"title":"Example Film","synopsis":"A detective investigates a violent murder while uncovering a dark family secret.","threshold":0.35}'
```

Search similar content:

```bash
curl -X POST http://127.0.0.1:8000/search-similar \
  -H "Content-Type: application/json" \
  -d '{"title":"Example Film","synopsis":"A detective investigates a violent murder while uncovering a dark family secret.","top_k":5}'
```

Combined intelligence report:

```bash
curl -X POST http://127.0.0.1:8000/intelligence-report \
  -H "Content-Type: application/json" \
  -d '{"title":"Example Film","synopsis":"A detective investigates a violent murder while uncovering a dark family secret.","top_k_similar":5}'
```

## Streamlit Demo

Ensure local artifacts exist first:

```bash
python scripts/train_baseline.py
python scripts/build_search_index.py
```

Run the demo:

```bash
python scripts/run_streamlit.py
```

The UI demonstrates:

- title and synopsis input
- predicted tags with confidence scores
- semantic similar-content recommendations
- structured metadata JSON
- model and artifact status

Screenshots can be added later under [docs/screenshots](docs/screenshots).

## Makefile Commands

```bash
make install
make test
make prepare-data
make eda
make train-baseline
make build-search-index
make predict-sample
make search-sample
make intelligence-sample
make run-api
make run-streamlit
```

## Documentation

- [Architecture](docs/architecture.md)
- [Model Card](docs/model_card.md)
- [Sample Outputs](docs/sample_outputs.md)
- [Portfolio Summary](docs/portfolio_summary.md)

## Tests And CI

Run tests locally:

```bash
pytest
```

GitHub Actions runs the lightweight test suite on push and pull request. CI does not require the raw dataset, trained model, embeddings, Hugging Face downloads or other local artifacts.

## Planned Work

- Docker and deployment packaging
- richer Streamlit polish and screenshots
- threshold calibration and per-tag diagnostics
- stronger text classification models
- production-grade vector search if scale requires it
