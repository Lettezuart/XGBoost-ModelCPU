import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# 1. Carrega les dades
df = pd.read_csv("Data/Raw/591/591_train.csv", header=None, sep=';', decimal=",")
  # Substitueix pel teu fitxer real
column_names = [
    "year", "month", "day", "hour", "minute", "second", "glucose_level", "finger_stick", "basal", "bolus",
    "sleep", "work", "stressors", "hypo_event", "illness",
    "exercise", "basis_heart_rate", "basis_gsr", "basis_skin_temperature",
    "basis_air_temperature", "basis_step", "basis_sleep", "meal", "type_of_meal"
]
df.columns = column_names
# 2. Separa entrada i sortida
X = df.drop(columns="glucose_level")  # Canvia 'target' pel nom correcte de la columna objectiu
y = df["glucose_level"]  # Canvia 'target' pel nom correcte de la columna objectiu

# 3. Divideix en train i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Converteix a DMatrix (entrenament amb GPU)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 5. Defineix paràmetres amb CUDA
params = {
    "tree_method": "hist",
    "predictor": "gpu_predictor",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "device": "cuda",
}

# 6. Crea watchlist per veure evolució del RMSE
watchlist = [(dtrain, "train"), (dtest, "eval")]

# 7. Entrena
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100000000,
    evals=watchlist,
    early_stopping_rounds=100,
    verbose_eval=True
)

# 8. Prediccions i mètrica
y_pred = model.predict(dtest)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE sobre el set de test: {rmse:.4f}")
