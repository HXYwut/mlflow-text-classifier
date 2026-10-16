from fastapi import FastAPI, HTTPException

from .model import predict
from .schemas import PredictionResponse, TextRequest

app = FastAPI(
    title="Text Topic Classifier (hockey vs space)",
    description="TF-IDF + LogisticRegression trained on 20 Newsgroups.",
    version="1.0.0",
)


@app.get("/health", summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, summary="Classify a text")
def classify(req: TextRequest) -> PredictionResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    label, proba = predict(text)
    return PredictionResponse(label=label, probability=round(proba, 4))
