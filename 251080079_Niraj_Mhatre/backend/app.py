# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 01:53:48 2025

@author: Niraj Mhatre
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt

# -------------------------
# App initialization
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load ML model safely
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------------
# Data models
# -------------------------
class Location(BaseModel):
    lat: float
    lng: float

class RouteRequest(BaseModel):
    source: Location
    destination: Location

# -------------------------
# Utility: Haversine distance
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# -------------------------
# Routes
# -------------------------
@app.get("/")
def health():
    return {"status": "SmartPath backend running"}

@app.post("/route")
def compute_route(data: RouteRequest):

    # 1️⃣ Distance
    distance_km = haversine(
        data.source.lat,
        data.source.lng,
        data.destination.lat,
        data.destination.lng
    )

    # 2️⃣ Features (MUST match training = 3 features)
    features = np.array([[1.0, 0.0, 0.0]])

    # 3️⃣ Predict speed
    raw_speed = float(model.predict(features)[0])

    # 4️⃣ Clamp speed to realistic bounds
    speed_kmph = max(15.0, min(60.0, raw_speed * 10))

    # 5️⃣ Travel time (minutes)
    travel_time_min = (distance_km / speed_kmph) * 60

    return {
        "route": [
            [data.source.lat, data.source.lng],
            [data.destination.lat, data.destination.lng]
        ],
        "distance_km": round(distance_km, 2),
        "predicted_speed_kmph": round(speed_kmph, 1),
        "travel_time_min": round(travel_time_min, 1)
    }
