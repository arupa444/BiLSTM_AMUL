"""
FastAPI application for BiLSTM sales prediction
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="BiLSTM Sales Prediction API",
    description="API for predicting sales using BiLSTM with Attention",
    version="1.0.0"
)

# Global variables for model and scalers
model = "models/bilstm_attention_final.h5"
feature_scaler = "models/feature_scaler.pkl"
target_scaler = "models/target_scaler.pkl"
SEQ_LEN = 60

# Feature columns (must match training)
FEATURES = [
    'stock_on_hand', 'intransit_qty', 'pending_po_qty', 'lead_time_days',
    'sin_doy', 'cos_doy', 'sin_dow', 'cos_dow', 'festival_flag',
    'branch_enc', 'sku_enc'
]

# Branch and SKU encoding mappings (you'll need to update these based on your training data)
BRANCH_MAPPING = {'GC01': 0, 'GC02': 1, 'GC03': 2, 'GC04': 3, 'GC05': 4}
SKU_MAPPING = {'SKU_A': 0, 'SKU_B': 1, 'SKU_C': 2, 'SKU_D': 3, 'SKU_E': 4}

# Festival dates (update based on your data)
FESTIVAL_DATES = ['2022-10-15', '2023-10-24', '2024-11-01']


class PredictionInput(BaseModel):
    """Input schema for single prediction"""
    branchcode: str = Field(..., example="GC01")
    materialcode: str = Field(..., example="SKU_A")
    stock_on_hand: float = Field(..., example=500.0)
    intransit_qty: float = Field(..., example=100.0)
    pending_po_qty: float = Field(..., example=50.0)
    lead_time_days: int = Field(..., example=7)
    date: str = Field(..., example="2024-01-15")


class HistoricalDataInput(BaseModel):
    """Input schema for prediction with historical sequence"""
    branchcode: str
    materialcode: str
    historical_data: List[PredictionInput] = Field(
        ...,
        min_items=60,
        description=f"Must provide exactly {SEQ_LEN} historical data points"
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    predicted_sales: float
    branchcode: str
    materialcode: str
    prediction_date: str


def load_models():
    """Load the trained model and scalers"""
    global model, feature_scaler, target_scaler

    try:
        # Update these paths to your actual model location
        model_path = "bilstm_attention_final.h5"
        feature_scaler_path = "feature_scaler.pkl"
        target_scaler_path = "target_scaler.pkl"

        # Load model
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Load scalers
        with open(feature_scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)

        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)

        print("✅ Model and scalers loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise


def is_festival_period(date_str: str, window: int = 7) -> int:
    """Check if date is within festival period"""
    date = pd.to_datetime(date_str)
    festivals = [pd.to_datetime(d) for d in FESTIVAL_DATES]

    for f in festivals:
        if abs((date - f).days) <= window:
            return 1
    return 0


def create_cyclical_features(date_str: str) -> dict:
    """Create cyclical date features"""
    date = pd.to_datetime(date_str)
    day_of_year = date.dayofyear
    weekday = date.weekday()

    return {
        'sin_doy': np.sin(2 * np.pi * day_of_year / 365.25),
        'cos_doy': np.cos(2 * np.pi * day_of_year / 365.25),
        'sin_dow': np.sin(2 * np.pi * weekday / 7),
        'cos_dow': np.cos(2 * np.pi * weekday / 7)
    }


def preprocess_single_input(input_data: PredictionInput) -> dict:
    """Preprocess single input data point"""
    # Encode branch and SKU
    if input_data.branchcode not in BRANCH_MAPPING:
        raise ValueError(f"Unknown branch code: {input_data.branchcode}")
    if input_data.materialcode not in SKU_MAPPING:
        raise ValueError(f"Unknown material code: {input_data.materialcode}")

    branch_enc = BRANCH_MAPPING[input_data.branchcode]
    sku_enc = SKU_MAPPING[input_data.materialcode]

    # Create cyclical features
    cyclical = create_cyclical_features(input_data.date)

    # Festival flag
    festival_flag = is_festival_period(input_data.date)

    # Combine all features
    features = {
        'stock_on_hand': input_data.stock_on_hand,
        'intransit_qty': input_data.intransit_qty,
        'pending_po_qty': input_data.pending_po_qty,
        'lead_time_days': input_data.lead_time_days,
        'sin_doy': cyclical['sin_doy'],
        'cos_doy': cyclical['cos_doy'],
        'sin_dow': cyclical['sin_dow'],
        'cos_dow': cyclical['cos_dow'],
        'festival_flag': festival_flag,
        'branch_enc': branch_enc,
        'sku_enc': sku_enc
    }

    return features


def create_sequence(data_points: List[dict]) -> np.ndarray:
    """Create sequence array from list of feature dictionaries"""
    # Convert to DataFrame
    df = pd.DataFrame(data_points)

    # Ensure correct order of features
    df = df[FEATURES]

    # Scale features
    scaled_features = feature_scaler.transform(df)

    # Reshape for LSTM input: (1, SEQ_LEN, n_features)
    sequence = scaled_features.reshape(1, SEQ_LEN, len(FEATURES))

    return sequence


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BiLSTM Sales Prediction API",
        "status": "active",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scalers_loaded": feature_scaler is not None and target_scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(input_data: HistoricalDataInput):
    """
    Predict sales based on historical sequence data

    Requires exactly 60 historical data points for the sequence
    """
    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Check if we have exactly SEQ_LEN data points
        if len(input_data.historical_data) != SEQ_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {SEQ_LEN} historical data points, got {len(input_data.historical_data)}"
            )

        # Preprocess all historical data points
        processed_data = []
        for data_point in input_data.historical_data:
            features = preprocess_single_input(data_point)
            processed_data.append(features)

        # Create sequence
        sequence = create_sequence(processed_data)

        # Make prediction
        prediction_scaled = model.predict(sequence, verbose=0)

        # Inverse transform prediction
        prediction = target_scaler.inverse_transform(prediction_scaled)[0, 0]

        # Ensure non-negative prediction
        prediction = max(0, prediction)

        return PredictionResponse(
            predicted_sales=float(prediction),
            branchcode=input_data.branchcode,
            materialcode=input_data.materialcode,
            prediction_date=input_data.historical_data[-1].date
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(inputs: List[HistoricalDataInput]):
    """
    Batch prediction endpoint
    """
    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    predictions = []

    for input_data in inputs:
        try:
            result = await predict_sales(input_data)
            predictions.append(result)
        except Exception as e:
            predictions.append({
                "error": str(e),
                "branchcode": input_data.branchcode,
                "materialcode": input_data.materialcode
            })

    return {"predictions": predictions}


@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "sequence_length": SEQ_LEN,
        "features": FEATURES,
        "n_features": len(FEATURES),
        "branches": list(BRANCH_MAPPING.keys()),
        "skus": list(SKU_MAPPING.keys()),
        "model_layers": len(model.layers),
        "model_parameters": model.count_params()
    }


if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(
        "main:app",  # Change "main" to your filename if different
        host="0.0.0.0",
        port=8000,
        reload=True
    )