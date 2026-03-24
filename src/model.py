import mlflow
import os
import numpy as np
import joblib

# Configurer le tracking URI (même chose que dans ton notebook)
# mlflow.set_tracking_uri("sqlite:///" + os.path.abspath("mlflow.db"))

def load_model():
    model = joblib.load("models/best_model.joblib")
    return model


def load_scaler():
    return joblib.load("models/scaler.joblib") 
