import io
import uvicorn
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from keras.models import load_model
import joblib

# ======================================================
# CONFIG
# ======================================================
API_KEY = "mysecretkey"


def verify_key(key: str):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True


# ======================================================
# LOAD ARTIFACTS
# ======================================================
MODEL_PATH = "models/bilstm_attention_final.h5"
FEATURE_SCALER_PATH = "models/feature_scaler.pkl"
TARGET_SCALER_PATH = "models/target_scaler.pkl"

model = load_model(MODEL_PATH)
feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)

# Detect model input shape
WINDOW = model.input_shape[1]
FEATURES = model.input_shape[2]

app = FastAPI(title="Forecasting API", description="LSTM/BiLSTM Forecasting Service")

# ======================================================
# DB Setup (SQLite)
# ======================================================
DB = "predictions.db"


def init_db():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecast_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            steps INTEGER,
            prediction TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def store_prediction(steps, prediction_list):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO forecast_history (steps, prediction) VALUES (?, ?)",
        (steps, str(prediction_list))
    )
    conn.commit()
    conn.close()


# ======================================================
# REQUEST MODELS
# ======================================================
class PredictRequest(BaseModel):
    data: list  # last window of input features


# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def predict_single_step(input_data):
    arr = np.array(input_data).reshape(1, WINDOW, FEATURES)
    arr_scaled = feature_scaler.transform(arr.reshape(-1, FEATURES)).reshape(1, WINDOW, FEATURES)
    pred = model.predict(arr_scaled)
    return float(target_scaler.inverse_transform(pred)[0][0])


def multi_step_forecast(input_data, n_steps):
    results = []
    window = np.array(input_data).reshape(1, WINDOW, FEATURES)

    for _ in range(n_steps):
        scaled = feature_scaler.transform(window.reshape(-1, FEATURES)).reshape(1, WINDOW, FEATURES)
        pred = model.predict(scaled)
        inv = target_scaler.inverse_transform(pred)[0][0]
        results.append(float(inv))

        # Update window (recursive)
        next_input = np.zeros((1, 1, FEATURES))
        next_input[0, 0, 0] = pred
        window = np.append(window[:, 1:, :], next_input, axis=1)

    return results


# ======================================================
# ENDPOINT 1: Single Step Prediction
# ======================================================
@app.post("/predict")
def predict(req: PredictRequest, valid=Depends(verify_key)):
    if len(req.data) != WINDOW:
        raise HTTPException(400, f"Expected window size {WINDOW}, got {len(req.data)}")
    pred = predict_single_step(req.data)
    store_prediction(1, [pred])
    return {"prediction": pred}


# ======================================================
# ENDPOINT 2: Predict N Steps
# ======================================================
@app.get("/predict_n")
def predict_n(steps: int, key: str = Depends(verify_key)):
    raise HTTPException(400, "Provide 'data' in body using POST /predict_n_body instead.")


@app.post("/predict_n_body")
def predict_n_body(req: PredictRequest, steps: int = 10, valid=Depends(verify_key)):
    preds = multi_step_forecast(req.data, steps)
    store_prediction(steps, preds)
    return {"steps": steps, "forecast": preds}


# ======================================================
# ENDPOINT 3: 7-Day Forecast
# ======================================================
@app.post("/predict_7_days")
def predict_7days(req: PredictRequest, valid=Depends(verify_key)):
    preds = multi_step_forecast(req.data, 7)
    store_prediction(7, preds)
    return {"7_day_forecast": preds}


# ======================================================
# ENDPOINT 4: Predict From CSV Upload
# ======================================================
@app.post("/predict_from_csv")
def predict_from_csv(file: UploadFile = File(...), steps: int = 7, key: str = Depends(verify_key)):
    df = pd.read_csv(file.file)

    if df.shape[0] < WINDOW:
        raise HTTPException(400, f"CSV must contain at least {WINDOW} rows")

    last_window = df.tail(WINDOW).values.tolist()
    preds = multi_step_forecast(last_window, steps)

    store_prediction(steps, preds)
    return {"forecast": preds}


# ======================================================
# ENDPOINT 5: Plot Forecast
# ======================================================
@app.post("/plot_forecast")
def plot_forecast(req: PredictRequest, steps: int = 30, key: str = Depends(verify_key)):
    preds = multi_step_forecast(req.data, steps)

    plt.figure(figsize=(10, 5))
    plt.plot(preds, label="Forecast")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


# ======================================================
# ENDPOINT 6: View Prediction History
# ======================================================
@app.get("/history")
def history(key: str = Depends(verify_key)):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM forecast_history ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    return {"history": rows}


# ======================================================
# START SERVER
# ======================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
