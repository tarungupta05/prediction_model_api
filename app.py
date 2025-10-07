import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import numpy as np
import joblib

# ----------------------------
# Environment Variables
# ----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "random_forest_model.pkl")
CUSTOM_THRESHOLD = float(os.getenv("CUSTOM_THRESHOLD", 0.2))

# ----------------------------
# Load Model Once at Startup
# ----------------------------
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at path: {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Random Forest API", version="1.0")

# ----------------------------
# Input Schema
# ----------------------------
class InputData(BaseModel):
    features: list[float]

# ----------------------------
# Error Handlers
# ----------------------------
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the traceback in server logs
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/predict_proba")
async def predict_proba(data: InputData):
    try:
        # Validate input
        if not data.features or not isinstance(data.features, list):
            raise HTTPException(status_code=400, detail="Invalid input format")

        # Convert to numpy array
        X = np.array(data.features).reshape(1, -1)

        # Predict probabilities
        proba = model.predict_proba(X)[0]
        prediction = int(proba[1] >= CUSTOM_THRESHOLD)

        return {
            "probability_class_0": float(proba[0]),
            "probability_class_1": float(proba[1]),
            "custom_threshold": CUSTOM_THRESHOLD,
            "prediction_based_on_threshold": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
