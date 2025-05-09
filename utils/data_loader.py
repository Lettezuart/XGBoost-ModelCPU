from utils.datasets import load_patient_data
from utils.data_processing import process_all_data

def load_and_prepare_data(base_path):
    test_data, train_data = load_patient_data(base_path)
    processed_test_data, processed_train_data = process_all_data(test_data, train_data)

    # Separar X i Y
    X_train = processed_train_data.drop(columns=['glucose_level', 'datetime'])
    Y_train = processed_train_data['glucose_level']

    X_test = processed_test_data.drop(columns=['glucose_level', 'datetime'])
    Y_test = processed_test_data['glucose_level']

    return X_train, Y_train, X_test, Y_test

