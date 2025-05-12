import numpy as np
from filterpy.kalman import KalmanFilter

def apply_kalman_to_data(df, patient_id=None):
    """
    Apply Kalman filter to the glucose_level column (zeros) in the DataFrame.
    For patient 591, apply AR extrapolation. For others, use causal Kalman.
    """
    df_filtered = df.copy()
    values = df_filtered["glucose_level"].astype(float).to_numpy()

    if patient_id == 591:
        # --- Dynamic AR extrapolation ---
        max_p = 5
        zero_positions = np.where(values == 0)[0]
        for pos in zero_positions:
            prev = []
            j = pos - 1
            while j >= 0 and len(prev) < max_p:
                if values[j] != 0:
                    prev.append((j, values[j]))
                j -= 1
            if len(prev) < 2:
                values[pos] = prev[0][1] if prev else 120.0
            else:
                idxs, ys = zip(*prev)
                x = np.array([pos - idx for idx in idxs])
                y = np.array(ys)
                coef = np.polyfit(x, y, 1)
                values[pos] = np.polyval(coef, 0)
        df_filtered["glucose_level"] = values
        return df_filtered

    # --- Standard Kalman filter ---
    kf = KalmanFilter(dim_x=1, dim_z=1)

    # Estimar estat inicial com el primer valor no-zero
    nonzero_start = next((v for v in values if v != 0), 120.0)
    kf.x = np.array([nonzero_start])
    kf.P *= 100.0
    kf.F = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.R = np.array([[1.0]])
    kf.Q = np.array([[0.1]])

    # Aplicar Kalman pas a pas
    for i in range(len(values)):
        if values[i] == 0:
            kf.predict()
            values[i] = float(kf.x)
        else:
            kf.predict()
            kf.update([values[i]])

        # Clamp (opcional, per evitar valors negatius o bojos)
        values[i] = max(0.0, min(values[i], 400.0))

    df_filtered["glucose_level"] = values
    return df_filtered


def apply_kalman_to_all_data(prepared_data):
    """
    Given a dict {patient_id: (df_train, df_test)},
    apply the filter only to df_train and return the same structure.
    """
    filtered = {}
    for pid, (df_tr, df_te) in prepared_data.items():
        df_tr_filt = apply_kalman_to_data(df_tr, pid)
        filtered[pid] = (df_tr_filt, df_te)
    return filtered
