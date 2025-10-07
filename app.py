import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
CUSTOM_THRESHOLD = float(os.getenv("CUSTOM_THRESHOLD", "0.2"))

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

app = FastAPI(title="Random Forest API")

class InputData(BaseModel):
    features: list

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_proba")
def predict_proba(data: InputData):
    if not isinstance(data.features, list) or len(data.features) == 0:
        raise HTTPException(status_code=400, detail="Invalid input format")

    X = np.array(data.features).reshape(1, -1)
    proba = model.predict_proba(X)[0]
    prediction = int(proba[1] >= CUSTOM_THRESHOLD)

    return {
        "probability_class_0": float(proba[0]),
        "probability_class_1": float(proba[1]),
        "custom_threshold": CUSTOM_THRESHOLD,
        "prediction_based_on_threshold": prediction
    }
