# Provenance

This file ties together code, data, and model for each run.

| Run | Git commit | Data version | DVC hash | Model artifact |
|---|---|---|---|---|
| v1-baseline | _(auto-logged as MLflow tag)_ | v1 | _(auto-logged)_ | `ml/registry/model.joblib` |
| v2-improved | _(auto-logged as MLflow tag)_ | v2 | _(auto-logged)_ | `ml/registry/model.joblib` |

To reproduce any run:

```bash
git checkout <commit>
dvc pull data/raw/<version>
python -m ml.train --data-version <v> --experiment <name>
```
