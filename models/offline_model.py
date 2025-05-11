from utils.datasets import load_patient_data
from utils.data_processing import process_all_data
from models.training import xgbTrain_model
from utils.evaluation import plot_glucose_timeseries
from utils.clarke import plot_clarke_error
from utils.filter import apply_kalman_to_data
from sklearn.metrics import mean_absolute_error, r2_score
from utils.data_loader import load_and_prepare_data
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

def shift_target(df, horizon):
    """Shifts glucose levels to predict 'horizon' minutes into the future"""
    steps = horizon // 5
    df = df.copy()
    df["glucose_target"] = df["glucose_level"].shift(-steps)
    return df.dropna()


def train_and_evaluate_offline(base_path):
    # Load and split the data for each patient
    prepared_data = load_and_prepare_data(base_path)

    # Apply the Kalman filter to both train and test sets for each patient
    prepared_data = {
        patient_id: (
            apply_kalman_to_data(df_train),
            apply_kalman_to_data(df_test)
        )
        for patient_id, (df_train, df_test) in prepared_data.items()
    }

    results = {30: {}, 60: {}}

    output_dir = "Evaluation_results"

    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create output directories for each patient
    os.makedirs(output_dir, exist_ok=True)

    for patient_id, (df_train, df_test) in prepared_data.items():
        print(f"\n📊 Processing patient: {patient_id}")

        # Convert categorical variable 'time_of_day' into dummy variables
        df_train = pd.get_dummies(df_train, columns=['time_of_day'])
        df_test = pd.get_dummies(df_test, columns=['time_of_day'])

        for horizon in [30, 60]:
            print(f"🕒 Prediction for {horizon} minutes ahead")

            # Shift the target column for the prediction horizon
            df_train_shifted = shift_target(df_train, horizon)
            df_test_shifted = shift_target(df_test, horizon)

            # Split into features (X) and target (Y)
            X_train = df_train_shifted.drop(columns=["glucose_level", "glucose_target", "datetime"])
            Y_train = df_train_shifted["glucose_target"]

            X_test = df_test_shifted.drop(columns=["glucose_level", "glucose_target", "datetime"])
            Y_test = df_test_shifted["glucose_target"]

            # Train the model and get predictions
            model, x_scaler, rmse, mae, r2, Y_pred = xgbTrain_model(
                X_train, Y_train, X_test, Y_test, patient_id, horizon
            )

            print(f"✅ {patient_id} | Horizon {horizon} min → RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}")
            results[horizon][patient_id] = {
                "rmse": rmse, "mae": mae, "r2": r2
            }

            # Generate and save plots
            datetimes = df_test_shifted["datetime"]
            timeseries_filename = os.path.join(output_dir, f"glucose_timeseries_{patient_id}_h{horizon}min.png")
            plot_glucose_timeseries(Y_test, Y_pred, df_test_shifted["datetime"], patient_id, horizon, filename=timeseries_filename)            
            print(f"📉 Time series saved at: {timeseries_filename}")

            clarke_filename = os.path.join(output_dir, f"clarke_{patient_id}_h{horizon}min.png")
            plot_clarke_error(Y_test, Y_pred, filename=clarke_filename)
            print(f"📈 Clarke error saved at: {clarke_filename}")

    # Compute average metrics across all patients
    for horizon in [30, 60]:
        all_rmses = [v["rmse"] for v in results[horizon].values()]
        all_maes = [v["mae"] for v in results[horizon].values()]
        all_r2s = [v["r2"] for v in results[horizon].values()]

        avg_rmse = sum(all_rmses) / len(all_rmses)
        avg_mae = sum(all_maes) / len(all_maes)
        avg_r2 = sum(all_r2s) / len(all_r2s)

        print(f"\n📈 Results for horizon {horizon} minutes:")
        for pid, metrics in results[horizon].items():
            print(f"  Patient {pid}: RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}, LogLoss = {metrics['logloss']:.2f}, R² = {metrics['r2']:.2f}")
        print(f"  ➕ AVERAGE RMSE: {avg_rmse:.2f}")
        print(f"  ➕ AVERAGE MAE: {avg_mae:.2f}")
        print(f"  ➕ AVERAGE R²: {avg_r2:.2f}")

    return results



