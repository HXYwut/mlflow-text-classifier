# Current production model

| Field | Value |
|---|---|
| Model | TF-IDF + LogisticRegression |
| Dataset version | v2 (headers/footers/quotes stripped) |
| MLflow experiment | `improved` |
| Run name | `v2-improved` |
| Training commit | _(fill in after first run)_ |
| Accuracy | _(fill in)_ |
| F1 | _(fill in)_ |
| Artifact path | `ml/registry/model.joblib` |

## Why this one

The v2 cleaned dataset removes noisy headers/footers/quoted reply text that
leak senders and list metadata, so the classifier learns actual topic words
rather than artifacts of the newsgroup format. The improved run also tunes
`C` and `ngram_range`; see `ml/README_experiments.md` for the comparison.
