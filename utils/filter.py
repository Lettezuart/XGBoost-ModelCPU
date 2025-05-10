import numpy as np
from filterpy.kalman import KalmanFilter

def apply_kalman_to_data(df):
    """
    Apply Kalman filter to the glucose level (equal to 0) data in the DataFrame.
    """
    df_filtered = df.copy()

    # Make sure the glucose_level column is numeric
    df_filtered["glucose_level"] = df_filtered["glucose_level"].astype(float)

    # Identify rows where glucose_level is 0
    zero_glucose_mask = df_filtered["glucose_level"] == 0

    # If there are no zero glucose values, return the original DataFrame
    if not zero_glucose_mask.any():
        return df_filtered

    # Define the Kalman filter
    kalman_filter = KalmanFilter(dim_x=1, dim_z=1)
    
    # Initialize the state vector
    kalman_filter.x = np.array([0])  # Estimació inicial
    kalman_filter.P *= 1000  # Covariància inicial gran per la incertesa

    # Define the state transition matrix
    kalman_filter.F = np.array([[1]])  # La dinàmica del sistema (assumim que és una constant)
    kalman_filter.H = np.array([[1]])  # Observació (la glucosa mesurada)

    # Define the measurement noise covariance
    kalman_filter.R = np.array([[1]])  # Soroll de mesura (potser hauríem de ajustar-ho)

    # Define the process noise covariance
    kalman_filter.Q = np.array([[0.1]])  # Soroll del procés

    for idx in df_filtered[zero_glucose_mask].index:
        # If the previous value is not available, use 0
        prev_value = df_filtered.loc[idx-1, "glucose_level"] if idx > 0 else 0
        
        # Define the measurement
        measurement = np.array([prev_value])

        # Actualize the Kalman filter with the measurement
        kalman_filter.predict()
        kalman_filter.update(measurement)

        # Update the DataFrame with the filtered value
        df_filtered.loc[idx, "glucose_level"] = kalman_filter.x[0]

    return df_filtered

def apply_kalman_to_all_data(prepared_data):
    """
    Given a dict {patient: (df_train, df_test)}, apply the Kalman filter
    only to the df_train part, and return the same structure.
    """
    filtered = {}
    for patient_id, (df_train, df_test) in prepared_data.items():
        df_train_filtered = apply_kalman_to_data(df_train)
        # leave df_test untouched
        filtered[patient_id] = (df_train_filtered, df_test)
    return filtered