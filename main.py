from typing import List

import fastapi
import uvicorn

import joblib

from pydantic import BaseModel

import numpy as np


# TODO: import the contracts from the models folder for each classifier
class PredictionRequest(BaseModel):
    Text: List[str]


class InferenceRun(BaseModel):
    PredictedClass: str
    PredictedConfidence: float


app = fastapi.FastAPI()
classifier = joblib.load("models/20newsgroups/trained_20newsgroups.joblib")  # joblib or whatever format
classes = classifier.classes_  # TODO : Make this more modular add type hinting on model import somehow


@app.get("/")
async def root():  # TODO: update so all available classifiers are predicted modular way in a nice table
    classifiers = classifier.steps
    return {"message": str(classifiers)}   # TODO: Simple html template to print classifiers nicely


@app.post("/predict/")
async def predict(request: PredictionRequest) -> List[InferenceRun]:
    predictions = classifier.predict_proba(request.Text)
    results = []
    for pred in predictions:
        best_class = classes[np.argmax(pred)]
        confidence = np.max(pred)
        results.append(InferenceRun(PredictedClass=best_class, PredictedConfidence=confidence))
    return results


# if __name__ == "__main__":
#     host = "127.0.0.1"
#     # host = "0.0.0.0"  # use when deployed and change the port to the required one based on deployment environment
#     uvicorn.run(app, port=8000, debug=True, host=host)
