import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title = "Traffic Speed Prediction API")

class PredictionRequest(BaseModel):
    sensor_id : int
    previous_speed : float

# Create an empty dictionary to store our models in it later on.
models = {}

# HOMEPAGE
@app.get("/")
def root():
    return {
        "message": "Traffic Speed Prediction API is running!",
        "total_models": len(models),
        "available_sensors": list(models.keys())[:5] + ["..."]
    }

# When the server starts, start loading models in the empty dictionary.
@app.on_event("startup")
def load_models():
    if not os.path.exists("sensor_models"):
        raise HTTPException(status_code=500,detail = "sensor_models folder not found!")

    model_files = [f for f in os.listdir("sensor_models")]
 
    for file in model_files:
        sensor_id = int(file.split("_")[1].split(".")[0])
        model_path = os.path.join("sensor_models", file)
        with open(model_path, "rb") as f:
            models[sensor_id] = pickle.load(f)

# Making predictions
@app.post("/predict")
def predict_speed(request: PredictionRequest):
    sensor_id = request.sensor_id

    if sensor_id not in models:
        raise HTTPException(status_code=404, detail=f"Sensor{sensor_id} model not found")
    
    model = models[sensor_id]
    input_data = [[request.previous_speed]]  # sklearn needs 2D array format as input
    predicted_speed = model.predict(input_data)[0]

    return {
        "sensor_id": sensor_id,
        "previous_speed": request.previous_speed,
        "predicted_speed": round(predicted_speed.item(),4)
     }