from typing import List

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    Text: List[str]
