# Portfolio Summary

## Two-Sentence Overview

Media Metadata Intelligence System is an NLP portfolio project that predicts multiple story/content tags from a movie title and synopsis, then retrieves semantically similar films from the MPST dataset. It combines a transparent classical multi-label baseline, embedding-based retrieval, API endpoints and an interactive Streamlit demo into one reproducible system.

## Technical Stack

Python, pandas, scikit-learn, sentence-transformers, FastAPI, Streamlit, pytest, pyarrow, joblib and GitHub Actions.

## Key Features

- reproducible data validation, preprocessing and EDA
- TF-IDF + OneVsRest Logistic Regression multi-label classifier
- multi-label evaluation with micro-F1, macro-F1 and Hamming loss
- semantic search with all-MiniLM-L6-v2 embeddings and cosine nearest neighbours
- deduplicated similar-content recommendations
- CLI scripts, FastAPI service and Streamlit demo
- lightweight CI that does not require private data or generated artifacts

## Measurable Result

The baseline classifier achieves `micro_f1 = 0.4106`, `macro_f1 = 0.1840` and `hamming_loss = 0.0525` on the MPST test split at a validation-selected threshold of `0.50`.

## Engineering Decisions

- Kept classification classical and explainable before moving to advanced models.
- Used the original train/validation/test split to avoid leakage.
- Separated tag prediction from semantic retrieval so each layer can be evaluated and improved independently.
- Cached local artifacts in API and Streamlit layers rather than retraining or rebuilding embeddings at runtime.
- Ignored raw data, processed data, model files and embedding indexes to keep the repository clean.

## What This Demonstrates To Employers

This project demonstrates practical ML engineering: dataset validation, reproducible pipelines, multi-label evaluation, semantic retrieval, service design, interactive demo building, artifact management, testing and CI. It also shows judgement – the system is honest about baseline limitations while still presenting a useful, end-to-end product workflow.
