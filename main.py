
import joblib
import pandas as pd
import os
import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = "models:/pulsar-rf-model/1"

try:
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    model_source = f"mlflow ({MODEL_URI})"
except Exception as e:
    print("MLflow load error:", e)
    model = joblib.load("model.joblib")
    model_source = "local (model.joblib)"

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
    return {"message": "ML API is running",
            "model_source": model_source
            }


@app.post("/predict")
def predict(features: PulsarFeatures):
    input_data = pd.DataFrame([features.model_dump()])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
