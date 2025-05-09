from utils.datasets import load_patient_data
from utils.data_processing import process_all_data
from models.training import train_model
from utils.evaluation import plot_glucose_timeseries
from utils.clarke import plot_clarke_error
from utils.filter import apply_kalman_to_all_data
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import os


def shift_target(df, horizon):
    """Shifts glucose levels to predict 'horizon' minutes into the future"""
    steps = horizon // 5
    df = df.copy()
    df["glucose_target"] = df["glucose_level"].shift(-steps)
    return df.dropna()


def train_and_evaluate_offline(base_path):
    test_data, train_data = load_patient_data(base_path)
    # Aplicar el filtre de Kalman a totes les dades
    test_data_filtrat, train_data_filtrat = apply_kalman_to_all_data(test_data, train_data)
    results = {30: {}, 60: {}}

    output_dir = os.path.join("ResultatsTemporals", "grafics_offline")
    os.makedirs(output_dir, exist_ok=True)

    for patient_id in train_data_filtrat.keys():
        print(f"\n📊 Processant pacient: {patient_id}")

        processed_test, processed_train = process_all_data(
            {patient_id: test_data_filtrat[patient_id]},
            {patient_id: train_data_filtrat[patient_id]}
        )

        df_train = processed_train[patient_id]
        df_test = processed_test[patient_id]

        df_train = pd.get_dummies(df_train, columns=['time_of_day'])
        df_test = pd.get_dummies(df_test, columns=['time_of_day'])

        for horizon in [30, 60]:
            print(f"🕒 Predicció a {horizon} minuts")

            df_train_shifted = shift_target(df_train, horizon)
            df_test_shifted = shift_target(df_test, horizon)

            X_train = df_train_shifted.drop(columns=["glucose_level", "glucose_target", "datetime"])
            Y_train = df_train_shifted["glucose_target"]

            X_test = df_test_shifted.drop(columns=["glucose_level", "glucose_target", "datetime"])
            Y_test = df_test_shifted["glucose_target"]

            model, x_scaler, rmse, Y_pred = train_model(X_train, Y_train, X_test, Y_test)
            mae = mean_absolute_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)

            print(f"✅ {patient_id} | Horizon {horizon} min → RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}")
            results[horizon][patient_id] = {"rmse": rmse, "mae": mae, "r2": r2}

            # Gràfics
            # ➕ Glucose Time Series
            datetimes = df_test_shifted["datetime"]  # Ens assegurem que estigui alineat
            timeseries_filename = os.path.join(output_dir, f"glucose_timeseries_{patient_id}_h{horizon}min.png")
            plot_glucose_timeseries(Y_test, Y_pred, datetimes, patient_id, horizon, filename=timeseries_filename)
            print(f"📉 Time series guardat a: {timeseries_filename}")

            # ➕ Clarke Error Grid
            clarke_filename = f"clarke_{patient_id}_h{horizon}min.png"
            plot_clarke_error(Y_test, Y_pred, filename=clarke_filename)
            print(f"📈 Clarke guardat a: {os.path.join(output_dir, clarke_filename)}")

    # Mitjanes finals
    for horizon in [30, 60]:
        all_rmses = [v["rmse"] for v in results[horizon].values()]
        all_maes = [v["mae"] for v in results[horizon].values()]
        all_r2s = [v["r2"] for v in results[horizon].values()]

        avg_rmse = sum(all_rmses) / len(all_rmses)
        avg_mae = sum(all_maes) / len(all_maes)
        avg_r2 = sum(all_r2s) / len(all_r2s)

        print(f"\n📈 Resultats horitzó {horizon} minuts:")
        for pid, metrics in results[horizon].items():
            print(f"  Pacient {pid}: RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}, R² = {metrics['r2']:.2f}")
        print(f"  ➕ PROMIG RMSE: {avg_rmse:.2f}")
        print(f"  ➕ PROMIG MAE: {avg_mae:.2f}")
        print(f"  ➕ PROMIG R²: {avg_r2:.2f}")

    return results
