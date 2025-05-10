import pandas as pd
import os

column_names = [
    "year", "month", "day", "hour", "minute", "second", "glucose_level", "finger_stick", "basal", "bolus",
    "sleep", "work", "stressors", "hypo_event", "illness",
    "exercise", "basis_heart_rate", "basis_gsr", "basis_skin_temperature",
    "basis_air_temperature", "basis_step", "basis_sleep", "meal", "type_of_meal"
]

def load_patient_data(base_path):
    """
    Read all data for training patients from CSV files in the given directory.
    """
    train_data = {}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv") and "_train" in file.lower():
                file_path = os.path.join(root, file)
                key = file.split("_")[0]  # Assumeix que el nom del fitxer comença amb ID del pacient

                try:
                    df = pd.read_csv(file_path, sep=';', header=None, decimal=",")
                    if df.shape[1] == len(column_names):
                        df.columns = column_names
                        train_data[key] = df
                    else:
                        print(f"[WARN] Unintended columns in {file_path}: {df.shape[1]} columns")
                except Exception as e:
                    print(f"[ERROR] Loading {file_path}: {e}")

    return train_data



