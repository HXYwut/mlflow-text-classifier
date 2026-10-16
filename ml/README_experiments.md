# Experiments

Two experiments are tracked in MLflow:

| Experiment | Run | Data | Hyperparams | Notes |
|---|---|---|---|---|
| `baseline` | `v1-baseline` | v1 (raw) | `C=1.0`, unigrams, max_features=20k | First end-to-end model |
| `improved` | `v2-improved` | v2 (cleaned) | `C=4.0`, bigrams, max_features=20k | Noisy headers/quotes removed; regularization tuned |

## What gets logged per run

- **Params**: `data_version`, `C`, `ngram_max`, `max_features`, `model`, `vectorizer`
- **Metrics**: `accuracy`, `f1`, `precision`, `recall`
- **Tags**: `git_commit` (code version), `data_dvc_hash` (data version)
- **Artifact**: the fitted sklearn pipeline (both as MLflow `model/` and as
  `ml/registry/model.joblib` for the API to load)

## Metric of choice: F1 (positive class = space)

This is a balanced two-class problem, so accuracy is fine — but F1 is more
robust if class balance ever drifts. We care most about not mislabelling
space content as hockey (and vice versa), and F1 weights both directions
symmetrically.

## Which run goes to production

The **`improved` / v2 run** is the production candidate:

1. Removing headers/footers/quotes means the classifier learns topical
   words (e.g. "orbit", "puck") rather than list-format artifacts
   (e.g. "Organization: University of X"), so it generalises better to
   text that isn't from the 20 Newsgroups mailing list itself.
2. Tuning `C` and adding bigrams captures phrases like "power play" /
   "space shuttle" that unigrams miss.

After running both experiments, fill in the actual numbers here and in
`ml/registry/current_model.md`.
