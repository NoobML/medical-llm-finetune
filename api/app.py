import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)


app = FastAPI()

class QuestionRequest(BaseModel):
    question: str


@app.post("/predict")
def predict_endpoint(request: QuestionRequest):
    # call your inference function here
    answer = "placeholder"  # replace with actual prediction
    return {"answer": answer}






