from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_glucose_timeseries(Y_test, Y_pred, datetimes, patient_id, horizon, filename=None):
    """
    Ploteja els valors reals i predites de glucosa al llarg del temps amb datetimes reals.
    """
    plt.figure(figsize=(12, 5), dpi=300)

    plt.plot(datetimes, Y_test, label="Real", color="red", linewidth=1, alpha=1.0)
    plt.plot(datetimes, Y_pred, label="Predicció", color="blue", linewidth=1, alpha=0.8)
    plt.fill_between(datetimes, Y_test, Y_pred, color='red', alpha=0.1, label="Error")

    # Ajustar els límits de l'eix X per fer-lo més ampliat
    plt.xlim([datetimes.min(), datetimes.max()])  # Pots ajustar aquest rang segons el que desitgis
    plt.title(f"{patient_id} - Predicció vs Real | Horitzó: {horizon} min")
    plt.xlabel("Temps")
    plt.ylabel("Glucosa (mg/dL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

