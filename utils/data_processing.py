from utils.datasets import load_patient_data
from utils.features import add_all_features

def process_all_data(train_data):
    """
    Apply all transformations to train data.
    """
    processed_train_data = {}
    
    for patient_id, df in train_data.items():
        df = add_all_features(df)  
        processed_train_data[patient_id] = df

    return processed_train_data
