import mlflow
import os
import numpy as np
import joblib

print(os.path.abspath("mlruns"))
print(os.path.abspath("data/scaler.joblib"))

# Configurer le tracking URI (même chose que dans ton notebook)
mlflow.set_tracking_uri("sqlite:///" + os.path.abspath("mlflow.db"))

def load_model():
    try:
        model = mlflow.pyfunc.load_model("models:/sleep-fatigue-lgbm@champion")
        return model
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        raise
    mlflow.pyfunc.load_model("models:/sleep-fatigue-lgbm@champion")


def load_scaler():
    scaler = joblib.load("data/scaler.joblib") 
