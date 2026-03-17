# app/api.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib
import numpy as np
from pydantic import BaseModel, validator
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    model_path = os.getenv("MODEL_PATH", "models/fraud_model.pkl")
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    # Cleanup on shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="Fraud Detection API",
    description="Credit card fraud detection using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    @validator("Amount")
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount must be positive")
        return v

@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict")
def predict(transaction: TransactionData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        # Prepare input
        data = np.array([[
            transaction.Time,
            transaction.V1, transaction.V2, transaction.V3,
            transaction.V4, transaction.V5, transaction.V6,
            transaction.V7, transaction.V8, transaction.V9,
            transaction.V10, transaction.V11, transaction.V12,
            transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18,
            transaction.V19, transaction.V20, transaction.V21,
            transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27,
            transaction.V28, transaction.Amount
        ]])

        # Predict
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        # Log prediction
        logger.info({
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "is_fraud": bool(prediction == 1)
        })

        return {
            "fraud_prediction": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "is_fraud": bool(prediction == 1),
            "risk_level": get_risk_level(float(probability)),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

def get_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    elif probability >= 0.5:
        return "MEDIUM"
    elif probability >= 0.3:
        return "LOW"
    else:
        return "SAFE"