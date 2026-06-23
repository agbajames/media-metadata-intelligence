# Sample Outputs

Sample input:

```text
Title: Example Film
Synopsis: A detective investigates a violent murder while uncovering a dark family secret.
```

## `scripts/predict_tags.py`

```json
{
  "title": "Example Film",
  "predicted_tags": [
    {
      "tag": "murder",
      "confidence": 0.8718275543470143
    },
    {
      "tag": "violence",
      "confidence": 0.5011021407163678
    }
  ],
  "metadata": {
    "model_version": "baseline_tfidf_logreg_v1",
    "threshold": 0.35,
    "input_fields": ["title", "synopsis"]
  }
}
```

## `scripts/search_similar.py`

Known top five semantic matches:

1. Watching the Detectives – 0.5927
2. Tightrope – 0.5833
3. Exposed – 0.5743
4. Blackmail – 0.5674
5. Law & Order – 0.5608

Representative JSON:

```json
{
  "query": {
    "title": "Example Film",
    "input_fields": ["title", "synopsis"]
  },
  "similar_content": [
    {
      "rank": 1,
      "imdb_id": "tt0472205",
      "title": "Watching the Detectives",
      "similarity_score": 0.5927,
      "tags": ["cult", "prank", "flashback"],
      "synopsis_source": "wikipedia",
      "short_synopsis_preview": "The film opens on a dark film noir black and white scene..."
    }
  ],
  "metadata": {
    "retrieval_model": "sentence-transformers/all-MiniLM-L6-v2",
    "index_type": "sklearn_nearest_neighbors_cosine",
    "top_k": 5
  }
}
```

## `scripts/intelligence_report.py`

```json
{
  "input": {
    "title": "Example Film",
    "synopsis": "A detective investigates a violent murder while uncovering a dark family secret."
  },
  "predicted_tags": [
    {
      "tag": "murder",
      "confidence": 0.8718275543470143
    },
    {
      "tag": "violence",
      "confidence": 0.5011021407163678
    }
  ],
  "similar_content": [
    {
      "rank": 1,
      "title": "Watching the Detectives",
      "similarity_score": 0.5927,
      "tags": ["cult", "prank", "flashback"]
    }
  ],
  "metadata": {
    "tagging_model": "baseline_tfidf_logreg_v1",
    "retrieval_model": "sentence-transformers/all-MiniLM-L6-v2",
    "threshold": 0.5,
    "top_k": 5
  }
}
```

## FastAPI `/predict-tags`

The response shape matches the CLI tag prediction output:

```json
{
  "title": "Example Film",
  "predicted_tags": [
    {"tag": "murder", "confidence": 0.8718275543470143},
    {"tag": "violence", "confidence": 0.5011021407163678}
  ],
  "metadata": {
    "model_version": "baseline_tfidf_logreg_v1",
    "threshold": 0.35,
    "input_fields": ["title", "synopsis"]
  }
}
```

## FastAPI `/search-similar`

The response includes deduplicated similar-content recommendations:

```json
{
  "query": {
    "title": "Example Film",
    "input_fields": ["title", "synopsis"]
  },
  "similar_content": [
    {"rank": 1, "title": "Watching the Detectives", "similarity_score": 0.5927},
    {"rank": 2, "title": "Tightrope", "similarity_score": 0.5833},
    {"rank": 3, "title": "Exposed", "similarity_score": 0.5743},
    {"rank": 4, "title": "Blackmail", "similarity_score": 0.5674},
    {"rank": 5, "title": "Law & Order", "similarity_score": 0.5608}
  ],
  "metadata": {
    "retrieval_model": "sentence-transformers/all-MiniLM-L6-v2",
    "index_type": "sklearn_nearest_neighbors_cosine",
    "top_k": 5
  }
}
```

## FastAPI `/intelligence-report`

The response combines predicted tags and similar-content recommendations in one JSON object:

```json
{
  "input": {
    "title": "Example Film",
    "synopsis": "A detective investigates a violent murder while uncovering a dark family secret."
  },
  "predicted_tags": [
    {"tag": "murder", "confidence": 0.8718275543470143},
    {"tag": "violence", "confidence": 0.5011021407163678}
  ],
  "similar_content": [
    {"rank": 1, "title": "Watching the Detectives", "similarity_score": 0.5927},
    {"rank": 2, "title": "Tightrope", "similarity_score": 0.5833},
    {"rank": 3, "title": "Exposed", "similarity_score": 0.5743},
    {"rank": 4, "title": "Blackmail", "similarity_score": 0.5674},
    {"rank": 5, "title": "Law & Order", "similarity_score": 0.5608}
  ],
  "metadata": {
    "tagging_model": "baseline_tfidf_logreg_v1",
    "retrieval_model": "sentence-transformers/all-MiniLM-L6-v2",
    "threshold": 0.5,
    "top_k_similar": 5
  }
}
```

## Streamlit Demo

The Streamlit demo presents the same combined intelligence workflow in an interactive UI. A user enters a title and synopsis, adjusts the tag threshold and number of similar movies, then views predicted tags, recommendations, structured JSON and artifact status.
