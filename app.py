import json
import pickle

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.json") as f:
    COLUMNS = json.load(f)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.get_dummies(pd.DataFrame([customer.model_dump()]))
    row = row.reindex(columns=COLUMNS, fill_value=0).astype(float)

    prob = float(model.predict_proba(row)[0][1])
    return {
        "churn_probability": round(prob, 4),
        "prediction": "will leave" if prob >= 0.5 else "will stay",
    }
