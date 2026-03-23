from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.model import load_model, load_scaler

app = FastAPI()

# Charger modèle et scaler au démarrage
model = load_model()
scaler = load_scaler()

# Définir le schéma d'input avec Pydantic
class PredictionInput(BaseModel):
    daily_screen_time_hours: float
    phone_usage_before_sleep_minutes: int
    sleep_duration_hours: float
    caffeine_intake_cups: int
    physical_activity_minutes: int
    notifications_received_per_day: int
    age: int
    screen_to_sleep_ratio: float  

@app.post("/predict")
def predict(input: PredictionInput):
    npa = np.asarray(list(input.model_dump().values())).reshape(1,8)
    npa_stand = scaler.transform(npa)
    y_pred = model.predict(npa_stand)
    return {"mental_fatigue_score": float(y_pred[0])}
