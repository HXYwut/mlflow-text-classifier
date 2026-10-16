# Deployment

We build Docker images automatically; no actual cloud deployment is wired up
(this matches the assignment: "No release process is requested").

## Pipelines

| Trigger | Job | Output |
|---|---|---|
| PR to `dev` / `staging` / `main` | `lint-test` + `docker-build` | blocks merge on failure |
| Push to `staging` | `build-staging-image` | image tagged `:staging-<sha>` |
| Push to `main` | `build-prod-image` | image tagged `:prod-<sha>` and `:latest` |

## Staging

On push to `staging`, the workflow:

1. Checks out the code.
2. Writes a `.env` from GitHub Secrets scoped to the `staging` environment
   (see repo → Settings → Environments → `staging`).
3. Builds and pushes to GHCR:
   `ghcr.io/<owner>/<repo>:staging-<commit-sha>`.

## Production

On push to `main`, the same flow runs against the `production` environment
secrets, producing:

- `ghcr.io/<owner>/<repo>:prod-<commit-sha>` (immutable)
- `ghcr.io/<owner>/<repo>:latest` (moving tag)

## Running locally

```bash
docker build -t classifier .
docker run -p 8000:8000 -e MODEL_PATH=/app/ml/registry/model.joblib classifier
curl http://127.0.0.1:8000/health
```

> The trained `model.joblib` is not committed to git (it's an MLflow
> artifact). For the image to actually serve predictions, copy the file
> from `ml/registry/` into the build context before `docker build`, or
> mount it at runtime:
> `docker run -p 8000:8000 -v $PWD/ml/registry:/app/ml/registry classifier`.
