import pandas as pd
import os
from utils.filter import apply_kalman_to_all_data

# Noms de les columnes
column_names = [
    "year", "month", "day", "hour", "minute", "second", "glucose_level", "finger_stick", "basal", "bolus",
    "sleep", "work", "stressors", "hypo_event", "illness",
    "exercise", "basis_heart_rate", "basis_gsr", "basis_skin_temperature",
    "basis_air_temperature", "basis_step", "basis_sleep", "meal", "type_of_meal"
]

def load_patient_data(base_path):
    """
    Funció per carregar les dades dels pacients a partir de fitxers CSV.
    Per cridar aquesta funció cal posar:

    Args:
        base_path = r'data/raw'  
        
    Returns:
       test_data, train_data = load_patient_data(base_path)
    """
    test_data = {}
    train_data = {}
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                key = file.split("_")[0]  # Tant per train com test

                try:
                    df = pd.read_csv(file_path, sep=';', header=None, decimal=",")
                    if df.shape[1] == len(column_names):
                        df.columns = column_names
                        if "_test" in file.lower():     # Punt 3 aplicat aquí
                            test_data[key] = df
                        elif "_train" in file.lower():  # Punt 3 aplicat aquí
                            train_data[key] = df
                    else:
                        print(f"[WARN] Columnes inesperades a {file_path}: {df.shape[1]} columnes")
                except Exception as e:
                    print(f"[ERROR] Carregant {file_path}: {e}")
    return test_data, train_data



