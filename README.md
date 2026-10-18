# Text Topic Classifier — hockey vs space

A small MLOps demo: a FastAPI service that classifies a news post as
`hockey` or `space`, trained on the 20 Newsgroups dataset.

| Layer | Choice |
|---|---|
| Model | TF-IDF + LogisticRegression (sklearn) |
| Data | 20 Newsgroups, filtered to `rec.sport.hockey` / `sci.space` |
| Data versioning | DVC (v1 raw, v2 cleaned) |
| Experiment tracking | MLflow |
| API | FastAPI + Uvicorn |
| Container | `python:3.11-slim` |
| CI/CD | GitHub Actions |

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. build data (v1 + v2)
python -m ml.data_pipeline

# 3. track data with DVC (first time)
dvc add data/raw/v1 data/raw/v2
git add data/raw/*.dvc .gitignore

# 4. train baseline + improved, tracked by MLflow
python -m mlflow.ui &          # optional, browse http://127.0.0.1:5000
python -m ml.train --data-version v1 --experiment baseline --C 1.0
python -m ml.train --data-version v2 --experiment improved --C 4.0 --ngram-max 2

# 5. run the API
uvicorn app.main:app --reload
# POST http://127.0.0.1:8000/predict  {"text": "the puck went into the net"}
```

## Branching & release flow

```
feature/*  →  dev  →  staging  →  main
```

| Branch | What happens on push / PR |
|---|---|
| `feature/*` | PR against `dev` triggers CI (lint + test + docker build) |
| `dev` | PR triggers CI; merge → push to `dev` |
| `staging` | PR triggers CI; merge → **builds & pushes staging Docker image** (uses `environment: staging` secrets) |
| `main` | PR triggers CI; merge → **builds & pushes production Docker image** (tags `:latest` and `:prod-<sha>`) |

A PR cannot be merged if any CI job fails.

## Layout

```
app/            FastAPI service
ml/             training, data pipeline, MLflow docs, model registry
data/raw/       v1 (raw) and v2 (cleaned) — tracked by DVC, not git
tests/          pytest suite
.github/workflows/ci.yml
Dockerfile
```

See `DEPLOYMENT.md` for staging/prod details and `ml/README_*.md` for the
data and experiment write-ups.

Demo note: PR workflow feature -> dev -> staging -> main.
