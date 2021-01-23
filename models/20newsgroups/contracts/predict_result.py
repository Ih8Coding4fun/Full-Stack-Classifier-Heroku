from pydantic import BaseModel
from typing import List


class InferenceRun(BaseModel):
    PredictedClass: str
    PredictedConfidence: float
