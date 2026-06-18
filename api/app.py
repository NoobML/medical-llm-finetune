import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)


app = FastAPI()

class QuestionRequest(BaseModel):
    question: str


@app.get('/home')
def home():
    return {'message': 'Medical LLM API is Running'}


@app.post("/predict")
def predict_endpoint(request: QuestionRequest):
    # call your inference function here
    answer = "placeholder"  # replace with actual prediction
    return {"answer": answer}

# @app.post('/predict')
# def predict_endpoint(request: QuestionRequest):
#     from src.inference import predict
#     answer = predict(request.question)
#     return {'answer': answer}






