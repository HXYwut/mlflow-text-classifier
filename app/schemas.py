from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Inference request body."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Raw news/post text to classify (hockey vs space).",
        examples=["The Maple Leafs won their third straight game last night!"],
    )


class PredictionResponse(BaseModel):
    """Inference response body."""

    label: str = Field(..., description="Predicted category: 'hockey' or 'space'.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted class.")
