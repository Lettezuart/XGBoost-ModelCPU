import os
import shutil
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.filter import apply_kalman_to_data
from utils.evaluation import plot_glucose_timeseries
from utils.clarke import plot_clarke_error
from utils.datasets import load_patient_data_test
from utils.data_loader import load_and_prepare_test_data


def shift_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    steps = horizon // 5
    df2 = df.copy()
    df2["glucose_target"] = df2["glucose_level"].shift(-steps)
    return df2.dropna()


def final_validation(base_path):
    """
    For each patient:
     - Load the preprocessed test DataFrame
     - Skip the first 60 minutes (12 rows)
     - Apply Kalman / EMA
     - One-hot encode & shift target
     - Load the trained XGBoost model + scaler
     - Make predictions on X_test
     - Compute metrics and save plot
    """
    test_data = load_and_prepare_test_data(base_path)
    
    output_dir = "Validation_plots"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    results = {30: {}, 60: {}}

    for patient_id, df in test_data.items():
        pid = int(patient_id)
        print(f"\n📊 Validating patient: {pid}")

        # --- Skip first 12 rows (60min) ---
        df = df.iloc[12:].reset_index(drop=True)

        # --- Apply Kalman filter or EMA ---
        df = apply_kalman_to_data(df, pid)

        # --- One-hot encoding ---
        df = pd.get_dummies(df, columns=["time_of_day"])

        for horizon in [30, 60]:
            print(f"🕒 Horizon {horizon} minutes")

            # --- Shift target ---
            df_shift = shift_target(df, horizon)

            # --- Split X / Y ---
            X_test = df_shift.drop(columns=["glucose_level", "glucose_target", "datetime"])
            Y_test = df_shift["glucose_target"]
            datetimes = df_shift["datetime"]

            # --- Load model + scaler ---
            model_dir   = os.path.join("Trained_model_outputs", f"patient_{pid}", f"horizon_{horizon}min", "Model_parameters")
            model_path  = os.path.join(model_dir, "model_param.json")
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"❌ Missing model or scaler for patient {pid}, horizon {horizon}")
                continue

            booster = xgb.Booster()
            booster.load_model(model_path)
            scaler = joblib.load(scaler_path)

            # --- Align features ---
            trained_feats = booster.feature_names
            for col in trained_feats:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test.reindex(columns=trained_feats, fill_value=0)

            # --- Scale & predict ---
            Xs = scaler.transform(X_test)
            dmat = xgb.DMatrix(Xs, feature_names=trained_feats)
            Y_pred = booster.predict(dmat)

            # --- Metrics ---
            rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
            mae  = mean_absolute_error(Y_test, Y_pred)
            r2   = r2_score(Y_test, Y_pred)
            print(f"✅ {pid} | H{horizon}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")
            results[horizon][pid] = {"rmse": rmse, "mae": mae, "r2": r2}

            # --- Save plots ---
            ts_file     = os.path.join(output_dir, f"timeseries_{pid}_h{horizon}.png")
            clarke_file = os.path.join(output_dir, f"clarke_{pid}_h{horizon}.png")
            plot_glucose_timeseries(Y_test, Y_pred, datetimes, pid, horizon, filename=ts_file)
            plot_clarke_error(Y_test, Y_pred, filename=clarke_file)

    return results
