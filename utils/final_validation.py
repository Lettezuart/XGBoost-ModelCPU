import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np
from utils.data_loader import load_and_prepare_data
from utils.clarke import plot_glucose_timeseries, plot_clarke_error
from utils.filter import apply_kalman_to_data
import shutil
from utils.data_loader import shift_target
import xgboost as xgb

def final_validation_test(new_test_data_dict, horizon):
    """
    Valida el model entrenat utilitzant noves dades de test.
    - new_test_data_dict: diccionari amb les dades de test per pacient {patient_id: df_test}
    - horizon: horitzó de predicció (30 o 60)
    """
    results = {}
    output_dir = f"Final_Test_Results_h{horizon}min"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    for patient_id, df_test in new_test_data_dict.items():
        print(f"\n🧪 Final test → Patient {patient_id}, Horizon {horizon}min")

        # Apply Kalman filter
        df_test = apply_kalman_to_data(df_test)

        # Encode 'time_of_day' again
        df_test = pd.get_dummies(df_test, columns=['time_of_day'])

        # Shift the target for prediction horizon
        df_test_shifted = shift_target(df_test, horizon)

        # Split features and target
        X_test = df_test_shifted.drop(columns=["glucose_level", "glucose_target", "datetime"])
        Y_test = df_test_shifted["glucose_target"]

        # Load model and scaler
        base_output_dir = os.path.join("outputs", f"patient_{patient_id}", f"horizon_{horizon}min", "Model_parameters")
        model_path = os.path.join(base_output_dir, "model_param.json")
        scaler_path = os.path.join(base_output_dir, "scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"❌ Model or scaler not found for patient {patient_id}, horizon {horizon}")
            continue

        model = xgb.Booster()
        model.load_model(model_path)
        x_scaler = joblib.load(scaler_path)

        # Scale features
        X_test_scaled = x_scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        # Predict
        dtest = xgb.DMatrix(X_test_scaled)
        Y_pred = model.predict(dtest)

        # Evaluate
        rmse = root_mean_squared_error(Y_test, Y_pred, squared=False)
        mae = mean_absolute_error(Y_test, Y_pred)
        r2 = r2_score(Y_test, Y_pred)

        print(f"📊 Patient {patient_id} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}")
        results[patient_id] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

        # Save plots
        timeseries_path = os.path.join(output_dir, f"glucose_timeseries_{patient_id}.png")
        clarke_path = os.path.join(output_dir, f"clarke_{patient_id}.png")

        plot_glucose_timeseries(Y_test, Y_pred, df_test_shifted["datetime"], patient_id, horizon, filename=timeseries_path)
        plot_clarke_error(Y_test, Y_pred, filename=clarke_path)

    return results

