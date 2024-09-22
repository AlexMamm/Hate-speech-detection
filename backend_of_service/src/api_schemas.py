from pydantic import BaseModel


class CommentRequest(BaseModel):
    text: str


class CommentPrediction(BaseModel):
    normal_probability: float
    hate_probability: float
    label: str