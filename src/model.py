import mlflow
import os
import numpy as np
import joblib

# Configurer le tracking URI (même chose que dans ton notebook)
mlflow.set_tracking_uri("sqlite:///" + os.path.abspath("mlflow.db"))

def load_model():
    try:
        model = mlflow.pyfunc.load_model("models:/sleep-model-lgbm@champion")
        return model
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        raise
    mlflow.pyfunc.load_model("models:/sleep-model-lgbm@champion")


def load_scaler():
    return joblib.load("data/scaler.joblib") 
