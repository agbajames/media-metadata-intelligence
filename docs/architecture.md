# Architecture

The Media Metadata Intelligence System takes a movie title and synopsis as input, predicts multi-label story tags, and retrieves semantically similar movies from the MPST dataset.

## Input And Preprocessing

The core input is:

- `title`
- `synopsis`

The preprocessing layer combines title and synopsis with a clear separator, lowercases the text, removes obvious control characters and normalises whitespace. Cleaning is intentionally conservative so the same text can support both classical ML features and embedding-based retrieval.

## Tag Prediction

The baseline tag model is a multi-label text classifier:

- Features – `TfidfVectorizer`
- Classifier – `OneVsRestClassifier`
- Base estimator – `LogisticRegression`
- Output – one confidence score per possible MPST tag

At inference time, scores are thresholded with a global threshold. The current validation-selected threshold is `0.50`.

## Evaluation

The model is evaluated on the original MPST split:

- `train` – model fitting only
- `val` – threshold tuning only
- `test` – final evaluation only

Metrics:

- micro-F1 – global precision/recall balance across all tag decisions
- macro-F1 – average F1 across tags, giving rare tags equal weight
- Hamming loss – fraction of individual tag decisions that are incorrect

## Semantic Retrieval

The retrieval layer uses:

- embedding model – `sentence-transformers/all-MiniLM-L6-v2`
- index – sklearn `NearestNeighbors`
- distance metric – cosine distance
- similarity score – `1 - cosine_distance`

The system embeds each movie's combined text and stores local search artifacts. At query time, it embeds the input text, retrieves more candidates than requested, deduplicates by `imdb_id` and normalised title, then returns the highest-similarity unique results.

## Interfaces

The project exposes the same core capabilities through several interfaces:

- CLI scripts for reproducible local workflows
- FastAPI endpoints for service-style access
- Streamlit demo for interactive portfolio presentation

## Artifact Lifecycle

Generated artifacts are local only and ignored by Git:

- raw dataset CSV
- processed parquet data
- trained model files
- embedding matrices
- nearest-neighbour indexes
- generated reports and figures

The repository contains code, tests, documentation and placeholders, but not private or heavyweight generated artifacts.
