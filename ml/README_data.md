# Data

## Source

[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/), as shipped by
`sklearn.datasets.fetch_20newsgroups`. We keep only two categories:

- `rec.sport.hockey` → label `0`
- `sci.space` → label `1`

Train/test split is the one provided by scikit-learn (random seed 42).

## v1 — raw, noisy

**Location:** `data/raw/v1/{train,test}.csv`

Built with `fetch_20newsgroups(remove=())` — i.e. headers, footers, and
quoted reply text are kept. This is "what a real inbox looks like":
Subject lines, `From:`/`Organization:` headers, and lines starting with
`>` all stay in.

## v2 — cleaned

**Location:** `data/raw/v2/{train,test}.csv`

Built with `fetch_20newsgroups(remove=("headers", "footers", "quotes"))`.
What changes v1 → v2:

- Removed `From:`, `Subject:`, `Organization:`, `Lines:` and other header
  fields (these leak list identity and sender, not topic).
- Removed the signature/footer block.
- Removed quoted lines (`>` and the attribution line that precedes them).

Rows whose body becomes empty after stripping are dropped in `train.py`.

## DVC tracking

```bash
dvc add data/raw/v1 data/raw/v2
git add data/raw/*.dvc .gitignore
git commit -m "data: add v1 and v2"
```

Remote storage (Azure blob) is configured separately:

```bash
dvc remote add -d azure azure://<container>/<path>
# authenticate via AZURE_STORAGE_CONNECTION_STRING / AZURE_STORAGE_ACCOUNT
git commit .dvc/config -m "dvc: add azure remote"
```

## Dataset version → commit mapping

| Dataset | DVC-tracked directory | First introduced in commit |
|---|---|---|
| v1 | `data/raw/v1/` | _(fill after first `dvc add`)_ |
| v2 | `data/raw/v2/` | _(fill after cleaning pass)_ |

Each MLflow run also tags `data_dvc_hash` (a content hash of the CSVs) so
we can answer "which exact data produced this model?".
