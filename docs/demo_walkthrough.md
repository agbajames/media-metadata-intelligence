# Demo Walkthrough

This walkthrough shows how to demo the Media Metadata Intelligence System from a clean local setup. It uses the existing Makefile commands where possible and assumes the MPST CSV has been placed locally.

## Demo Objective

Demonstrate an end-to-end metadata intelligence workflow:

- start with a movie title and synopsis
- predict likely story/content tags with confidence scores
- retrieve semantically similar movies
- inspect structured JSON output
- show the same capabilities through CLI, FastAPI and Streamlit

The classifier is the TF-IDF + OneVsRest Logistic Regression baseline. SentenceTransformer embeddings are used for semantic search and recommendation.

## Required Local Artifacts

The raw MPST dataset must be available at:

```text
data/raw/mpst_full_data.csv
```

The demo generates these local artifacts:

- `data/processed/mpst_processed.parquet`
- `data/processed/tag_classes.json`
- `models/baseline_tfidf_logreg.joblib`
- `artifacts/search/mpst_embeddings.npy`
- `artifacts/search/mpst_search_metadata.parquet`
- `artifacts/search/mpst_nn_index.joblib`

These files are ignored by Git and should not be committed.

## Step 1 – Prepare Data

```bash
make prepare-data
```

This loads and validates the MPST CSV, parses tags, creates `combined_text`, and writes the processed parquet file.

## Step 2 – Train Baseline Tag Model

```bash
make train-baseline
```

This trains the TF-IDF + OneVsRest Logistic Regression baseline on the original MPST `train` split, tunes the global threshold on `val`, and evaluates on `test`.

Key result to mention:

- selected threshold – `0.50`
- micro-F1 – `0.4106`
- macro-F1 – `0.1840`
- Hamming loss – `0.0525`

## Step 3 – Build Semantic Search Index

```bash
make build-search-index
```

This embeds movie text with `sentence-transformers/all-MiniLM-L6-v2`, builds a sklearn cosine nearest-neighbour index, and saves the search artifacts locally.

## Step 4 – Run CLI Prediction

```bash
make predict-sample
```

This predicts tags for the sample detective/murder synopsis. The expected tag themes are `murder` and `violence`.

## Step 5 – Run Semantic Search

```bash
make search-sample
```

This retrieves deduplicated similar-content recommendations. Known top results for the sample are:

1. Watching the Detectives – 0.5927
2. Tightrope – 0.5833
3. Exposed – 0.5743
4. Blackmail – 0.5674
5. Law & Order – 0.5608

## Step 6 – Run Combined Intelligence Report

```bash
make intelligence-sample
```

This combines tag prediction and semantic retrieval in one structured JSON object. Use this step to explain how the system can support metadata enrichment or recommendation workflows.

## Step 7 – Run FastAPI

```bash
make run-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

Suggested endpoints to test:

- `GET /health`
- `POST /predict-tags`
- `POST /search-similar`
- `POST /intelligence-report`

Use the sample input:

```json
{
  "title": "Example Film",
  "synopsis": "A detective investigates a violent murder while uncovering a dark family secret."
}
```

## Step 8 – Run Streamlit

```bash
make run-streamlit
```

Open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

Use the UI to enter the sample title and synopsis, adjust the threshold, inspect the predicted tags, review similar movies, and show the structured metadata JSON.

## Suggested Interview Talking Points

- The original problem is multi-label metadata tagging from short movie text.
- The baseline classifier is deliberately transparent: TF-IDF features plus OneVsRest Logistic Regression.
- The evaluation keeps the original train/validation/test split to avoid leakage.
- Macro-F1 is weaker because rare MPST tags are difficult and class imbalance is substantial.
- Semantic search is a separate layer using SentenceTransformer embeddings, not a transformer classifier.
- FastAPI and Streamlit show how model logic can be exposed through service and demo interfaces.
- Generated artifacts are local only and ignored by Git, which keeps the repository clean.

## Troubleshooting

If the raw dataset is missing, place it at:

```text
data/raw/mpst_full_data.csv
```

If tag prediction artifacts are missing, run:

```bash
make train-baseline
```

If search artifacts are missing, run:

```bash
make build-search-index
```

If Hugging Face warns about unauthenticated requests, set `HF_TOKEN` in your environment. The warning is not fatal, but authentication can improve download reliability.

If Streamlit suggests installing Watchdog, treat it as optional local-performance advice rather than a required dependency.
