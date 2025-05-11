# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import shap
# import matplotlib.pyplot as plt
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import root_mean_squared_error

# # 1. Load the dataset
# df = pd.read_csv("Data/Raw/591/591_train.csv", header=None, sep=';', decimal=",")
# column_names = [
#     "year", "month", "day", "hour", "minute", "second", "glucose_level", "finger_stick", "basal", "bolus",
#     "sleep", "work", "stressors", "hypo_event", "illness",
#     "exercise", "basis_heart_rate", "basis_gsr", "basis_skin_temperature",
#     "basis_air_temperature", "basis_step", "basis_sleep", "meal", "type_of_meal"
# ]
# df.columns = column_names

# # 2. Split features and target
# X = df.drop(columns="glucose_level")
# y = df["glucose_level"]

# # 3. Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Convert to DMatrix
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)

# # 5. Define model parameters
# params = {
#     "tree_method": "hist",
#     "predictor": "gpu_predictor",
#     "objective": "reg:squarederror",
#     "eval_metric": "rmse",
#     "device": "cuda",
# }

# # 6. Set up watchlist
# watchlist = [(dtrain, "train"), (dtest, "eval")]

# # 7. Train the model
# model = xgb.train(
#     params=params,
#     dtrain=dtrain,
#     num_boost_round=10000,
#     evals=watchlist,
#     early_stopping_rounds=100,
#     verbose_eval=True
# )

# # 8. Make predictions and evaluate
# y_pred = model.predict(dtest)
# rmse = root_mean_squared_error(y_test, y_pred)
# print(f"Test RMSE: {rmse:.4f}")

# # 9. SHAP - Feature importance visualization
# print("Generating SHAP visualizations...")

# # Create output folder if it doesn't exist
# output_dir = "Shap"
# os.makedirs(output_dir, exist_ok=True)

# # Initialize SHAP explainer and calculate SHAP values
# explainer = shap.Explainer(model, X_train)
# shap_values = explainer(X_test)

# # SHAP summary plot
# summary_plot_path = os.path.join(output_dir, "shap_summary_plot.png")
# shap.summary_plot(shap_values, X_test, show=False)
# plt.savefig(summary_plot_path, bbox_inches="tight")
# plt.clf()
# print(f"SHAP summary plot saved to: {summary_plot_path}")

# # SHAP bar plot
# bar_plot_path = os.path.join(output_dir, "shap_summary_bar_plot.png")
# shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
# plt.savefig(bar_plot_path, bbox_inches="tight")
# plt.clf()
# print(f"SHAP summary bar plot saved to: {bar_plot_path}")

import os
import pandas as pd
from tabulate import tabulate

# Carpeta principal on hi ha subcarpetes per pacient
carpeta_principal = "Data/Raw"  # 🔁 Canvia això pel teu path

resultats = []

# Camina per totes les subcarpetes i fitxers
for root, dirs, files in os.walk(carpeta_principal):
    for fitxer in files:
        if fitxer.endswith(".csv"):
            cami_complet = os.path.join(root, fitxer)
            try:
                df = pd.read_csv(cami_complet, sep=';')  # ⛏️ Canvia el separador si cal
                if df.shape[1] >= 7:
                    col7 = df.iloc[:, 6]
                    zeros = (col7 == 0).sum()
                    resultats.append([fitxer, os.path.relpath(cami_complet, carpeta_principal), zeros])
                else:
                    resultats.append([fitxer, os.path.relpath(cami_complet, carpeta_principal), "❌ <7 columnes"])
            except Exception as e:
                resultats.append([fitxer, os.path.relpath(cami_complet, carpeta_principal), f"❌ Error: {str(e)}"])

# Mostrar resultats en taula
headers = ["Fitxer", "Ruta Relativa", "Zeros a la 7a columna"]
print(tabulate(resultats, headers=headers, tablefmt="grid"))

# (Opcional) Guardar la taula en un fitxer .txt
output_path = os.path.join(carpeta_principal, "resum_zeros_col7.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(tabulate(resultats, headers=headers, tablefmt="grid"))

print(f"\n✅ Resultats guardats a: {output_path}")


