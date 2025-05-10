from sklearn.model_selection import train_test_split
import pandas as pd
from utils.datasets import load_patient_data
from utils.data_processing import process_all_data


def load_and_prepare_data(base_path, test_size=0.2):
    """
    Load patient CSVs, apply all feature engineering (including datetime),
    then split each patient's data into train/test by time (no shuffle).
    Returns a dict: {patient_id: (df_train, df_test)}
    """
    raw_data = load_patient_data(base_path)
    prepared_data = {}

    for patient_id, df in raw_data.items():
        # 1) Apply ALL preprocessing (datetime + features)
        processed = process_all_data({patient_id: df})[patient_id]

        # 2) Now split chronologically (shuffle=False) into train/test
        df_train, df_test = train_test_split(
            processed,
            test_size=test_size,
            shuffle=False,
            random_state=42,
        )

        prepared_data[patient_id] = (df_train, df_test)

    return prepared_data



