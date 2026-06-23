# Model Card – Baseline Tag Classifier

## Model Name

TF-IDF + OneVsRest Logistic Regression baseline.

## Intended Use

The model predicts likely MPST story/content tags for a movie title and synopsis. It is intended to support metadata enrichment, catalogue exploration, recommendation prototypes and portfolio demonstration.

## Problem Type

Multi-label text classification.

## Input Fields

- `title`
- `synopsis`

The fields are combined into one conservative text representation before feature extraction.

## Output Format

The model returns a confidence score for each tag. Application layers apply a threshold and return JSON records such as:

```json
{"tag": "murder", "confidence": 0.87}
```

## Training Data

MPST Movie Plot Synopses with Tags, using the dataset's original `train` split. The raw data is expected locally at `data/raw/mpst_full_data.csv` and is not committed to the repository.

## Evaluation Data

Validation is used only for threshold selection. Final metrics are computed on the MPST `test` split.

## Metrics

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

## Known Limitations

- Macro-F1 is low because rare tags are difficult and the dataset is imbalanced.
- The classifier is a classical baseline, not a fine-tuned transformer classifier.
- Plot synopses may contain spoilers, biased summaries or inconsistent writing styles.
- Tags are inherited from the dataset and may be subjective or incomplete.
- A global threshold is simple and explainable, but may not be optimal for every tag.

## Ethical And UX Considerations

Predictions should support metadata and recommendation workflows, not be treated as perfect content truth. User-facing systems should make uncertainty visible, avoid overconfident language and allow human review for high-impact catalogue decisions.

## When Not To Use The Model

Do not use this model as the sole basis for age ratings, safety classification, legal compliance, moderation decisions or definitive content labelling. It is a portfolio baseline and prototype component.

## Future Improvements

- per-tag threshold calibration
- stronger imbalance handling
- richer error analysis for rare tags
- transformer-based classifier comparison
- confidence calibration
- human-in-the-loop metadata review workflow
