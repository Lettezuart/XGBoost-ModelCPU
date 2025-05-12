from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_glucose_timeseries(Y_test, Y_pred, datetimes, patient_id, horizon, filename=None):
    """
    Plots the actual vs predicted glucose levels over time.
    """
    plt.figure(figsize=(12, 5), dpi=300)

    plt.plot(datetimes, Y_test, label="Real", color="red", linewidth=1, alpha=1.0)
    plt.plot(datetimes, Y_pred, label="Predicted", color="blue", linewidth=1, alpha=0.8)
    plt.fill_between(datetimes, Y_test, Y_pred, color='red', alpha=0.1, label="Error")

    # Adjsuting the x-axis limits to show the full range of datetimes
    plt.xlim([datetimes.min(), datetimes.max()]) 
    plt.title(f"{patient_id} - Predicted vs Real | Horizon: {horizon} min")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

