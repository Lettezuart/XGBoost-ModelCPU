from utils.datasets import load_patient_data
from utils.features import add_all_features

def process_all_data(test_data, train_data):
    """Aplica totes les transformacions a les dades (test i train) per cada pacient."""
    processed_test_data = {}
    processed_train_data = {}

    # Processar les dades de test
    for patient_id, df in test_data.items():
        df = add_all_features(df)  # Aplica totes les transformacions
        processed_test_data[patient_id] = df

    # Processar les dades d'entrenament
    for patient_id, df in train_data.items():
        df = add_all_features(df)  # Aplica totes les transformacions
        processed_train_data[patient_id] = df

    return processed_test_data, processed_train_data
