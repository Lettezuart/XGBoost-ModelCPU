import numpy as np
from filterpy.kalman import KalmanFilter

def apply_kalman_to_data(df):
    """
    Apply Kalman filter to the glucose level (equal to 0) data in the DataFrame.
    """
    df_filtered = df.copy()

    # Assegurar-nos que la columna és de tipus float
    df_filtered["glucose_level"] = df_filtered["glucose_level"].astype(float)

    # Identificar els valors on glucosa és 0
    zero_glucose_mask = df_filtered["glucose_level"] == 0

    # Si no hi ha cap valor zero, no apliquem el filtre
    if not zero_glucose_mask.any():
        return df_filtered

    # Definir el filtre de Kalman
    kalman_filter = KalmanFilter(dim_x=1, dim_z=1)
    
    # Inicialitzar el filtre
    kalman_filter.x = np.array([0])  # Estimació inicial
    kalman_filter.P *= 1000  # Covariància inicial gran per la incertesa

    # Definir la matriu d'estat
    kalman_filter.F = np.array([[1]])  # La dinàmica del sistema (assumim que és una constant)
    kalman_filter.H = np.array([[1]])  # Observació (la glucosa mesurada)

    # Matriu de covariància de la mesura
    kalman_filter.R = np.array([[1]])  # Soroll de mesura (potser hauríem de ajustar-ho)

    # Matriu de covariància de control
    kalman_filter.Q = np.array([[0.1]])  # Soroll del procés

    for idx in df_filtered[zero_glucose_mask].index:
        # Si el valor anterior no és zero, utilitzem-lo com a valor inicial
        prev_value = df_filtered.loc[idx-1, "glucose_level"] if idx > 0 else 0
        
        # Definir la mesura
        measurement = np.array([prev_value])

        # Actualitzar el filtre amb la mesura
        kalman_filter.predict()
        kalman_filter.update(measurement)

        # Substituir el valor de glucosa per l'estimació del filtre
        df_filtered.loc[idx, "glucose_level"] = kalman_filter.x[0]

    return df_filtered

def apply_kalman_to_all_data(test_data, train_data):
    """
    Applies configurated Kalman filter to all data
    """
    test_data_filtrat = {patient: apply_kalman_to_data(df) for patient, df in test_data.items()}
    train_data_filtrat = {patient: apply_kalman_to_data(df) for patient, df in train_data.items()}
    
    return test_data_filtrat, train_data_filtrat