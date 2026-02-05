from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict
from typing import List

app = FastAPI(title="Student Pass Prediction API")

class StudentInput(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "Student Pass Prediction API is running"}

@app.post("/predict")
def predict_student(data: StudentInput):
    result = predict(data.features)
    return {"prediction": result}
