from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("traffic_prediction.joblib")

# Load sensor locations
sensors_df = pd.read_csv("graph_sensor_locations.csv")


class RouteRequest(BaseModel):
    source_lat: float
    source_lng: float
    dest_lat: float
    dest_lng: float
    hour: int
    day: int


def nearest_sensor(lat, lng):
    """
    Returns sensor_id of the closest sensor (using squared distance)
    """
    dist_sq = (
        (sensors_df["latitude"] - lat) ** 2 +
        (sensors_df["longitude"] - lng) ** 2
    )
    return int(sensors_df.loc[dist_sq.idxmin(), "sensor_id"])


@app.post("/route")
def route(req: RouteRequest):
    # Call OSRM
    osrm_url = (
        f"http://router.project-osrm.org/route/v1/driving/"
        f"{req.source_lng},{req.source_lat};"
        f"{req.dest_lng},{req.dest_lat}"
        "?overview=full&geometries=geojson"
    )

    response = requests.get(osrm_url).json()
    route_data = response["routes"][0]

    distance_km = route_data["distance"] / 1000
    geometry = route_data["geometry"]

    coords = geometry["coordinates"]

    #reduce compute
    sampled = coords[::5] if len(coords) > 50 else coords

    segment_km = distance_km / len(sampled)
    total_time_min = 0.0

    hour = req.hour
    day = req.day
    
    time_sin = np.sin(2 * np.pi * hour / 2)
    time_cos = np.cos(2 * np.pi * hour / 24)

    day_sin = np.sin(2 * np.pi * day / 7)
    day_cos = np.cos(2 * np.pi * day / 7)
    
    is_rush_hour = int(
        (7 <= hour <= 10) or (16 <= hour <= 19)
    )

    is_weekend = int(day == 6 or day == 0)
    
    for lng, lat in sampled:
        sensor_id = nearest_sensor(lat, lng)

        X = pd.DataFrame(
            [[lat, lng,
            time_sin, time_cos,
            day_sin, day_cos,
            is_rush_hour,
            is_weekend,]],
            columns=["latitude","longitude","time_sin","time_cos","day_sin","day_cos","is_rush_hour","is_weekend"]
        )

        speed_kmph = max(model.predict(X)[0], 5)
        total_time_min += (segment_km / speed_kmph) * 60

    return {
        "distance_km": round(distance_km, 2),
        "time_min": round(total_time_min, 2),
        "geometry": geometry
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r") as f:
        return f.read()
