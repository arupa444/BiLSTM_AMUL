from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import uvicorn
import json

app = FastAPI(title="Time Series Forecast API",
              description="Predict using LSTM/BiLSTM/Attention Model",
              version="1.0.0")

# -------------------------------------
# Load model + scaler + config
# -------------------------------------
MODEL_PATH = "model/bilstm_attention_final.h5"
SCALER_PATH = "model/scaler.pkl"
CONFIG_PATH = "model/config.json"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

WINDOW_SIZE = config.get("window_size", 20)   # default 20 timesteps


# -------------------------------------
# Request Schema
# -------------------------------------
class InputData(BaseModel):
    values: list   # last N timesteps (same as window_size)


# -------------------------------------
# Prediction Route
# -------------------------------------
@app.post("/predict")
def predict(data: InputData):

    if len(data.values) != WINDOW_SIZE:
        return {
            "error": f"Expected {WINDOW_SIZE} timesteps, got {len(data.values)}"
        }

    # Convert to np array
    X = np.array(data.values).reshape(-1, 1)

    # Scale the input
    X_scaled = scaler.transform(X)

    # Reshape for LSTM: (1, timesteps, features)
    X_scaled = X_scaled.reshape(1, WINDOW_SIZE, 1)

    # Run model prediction
    pred_scaled = model.predict(X_scaled)
    pred = scaler.inverse_transform(pred_scaled)[0][0]

    return {
        "forecast": float(pred)
    }


@app.get("/")
def home():
    return {
        "message": "Time Series Forecast API is running!",
        "window_size": WINDOW_SIZE
    }


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=8000)
