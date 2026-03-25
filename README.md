# Sleep_score 

## Problem statement
Avec la montée en puissance des écrans et la fragmentation de l'attention, 
la fatigue mentale est devenue un enjeu croissant du quotidien. 
Ce projet prédit le score de fatigue mentale d'un utilisateur à partir 
de ses habitudes (temps d'écran, sommeil, caféine, activité physique) 
via une API REST déployée dans Docker.

## Dataset
[Sleep, Screen Time and Stress Analysis](https://www.kaggle.com/datasets/jayjoshi37/sleep-screen-time-and-stress-analysis)
— 15 000 entrées, 13 colonnes, cible : `mental_fatigue_score` (régression, score 1-10)

## Approche
- Features utilisées : daily_screen_time_hours, phone_usage_before_sleep_minutes, sleep_duration_hours, caffeine_intake_cups, physical_activity_minutes, notifications_received_per_day, age
]
- Feature engineerée : `screen_to_sleep_ratio` (ratio temps d'écran / durée de sommeil)
- Exclusions : `stress_level` et `sleep_quality_score` (voir Limites)
- Tracking des expériences : MLflow avec SQLite backend
- Monitoring : Evidently pour la détection de data drift

## Résultats
| Modèle | RMSE | R² |
|--------|------|-----|
| Dummy baseline | 2.73 | 0 |
| Linear Regression | 1.47 | 0.71 |
| Random Forest | 1.44 | 0.72 |
| XGBoost | 1.48 | 0.71 |
| LightGBM ✓ | 1.42 | 0.73 |

## Lancer le projet

### Avec Docker
docker build -t sleep-fatigue-api .
docker run -p 8000:8000 sleep-fatigue-api

Swagger UI disponible sur http://localhost:8000/docs

### En local
pip install -r requirements.txt
uvicorn src.main:app --reload

## Structure
```
sleep_score/
│
├── src/
│   ├── main.py          # API FastAPI
│   └── model.py         # Chargement modèle et scaler
│
├── models/
│   ├── best_model.joblib
│   └── scaler.joblib
│
├── data/                # Non versionné (.gitignore)
│
├── eda.ipynb            # Exploration et preprocessing
├── modeling.ipynb       # Entraînement et MLflow tracking
├── monitoring.ipynb     # Data drift avec Evidently
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Limites
Dataset synthétique de 15k entrées — les prédictions reflètent des patterns simulés, pas des données cliniques réelles
j'ai exclu deux variables stress_level et sleep_quality_score car elle etait très correller avec la cible et sans vision sur leur creation, une erreur de data leakage aurait pu arriver