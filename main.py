
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

model = joblib.load("model.joblib")


class PulsarFeatures(BaseModel):
    mean_integrated_profile: float
    std_integrated_profile: float
    kurtosis_integrated_profile: float
    skewness_integrated_profile: float
    mean_dm_snr_curve: float
    std_dm_snr_curve: float
    kurtosis_dm_snr_curve: float
    skewness_dm_snr_curve: float


@app.get("/")
def home():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(features: PulsarFeatures):
    input_data = pd.DataFrame([features.model_dump()])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
