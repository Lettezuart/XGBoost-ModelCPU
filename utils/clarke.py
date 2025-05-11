import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error

# Funció per calcular el RMSE específic de glucosa (gRMSE)
def glucose_rmse(act_vals, pred_vals):
    mse = mean_squared_error(act_vals, pred_vals)
    rmse = np.sqrt(mse)
    return rmse

# Càlcul de l'R²
def r2_score_custom(y_true, y_pred):
    """
    Calculate R² score between the actual and predicted values.
    """
    return r2_score(y_true, y_pred)

# Clarke Error Grid detailed function
def clarke_error_zone_detailed(act, pred):
    """
    This function outputs the Clarke Error Grid region (encoded as integer)
    for a combination of actual and predicted value
    Based on 'Evaluating clinical accuracy of systems for self-monitoring of blood glucose'
    """
    # Zone A
    if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
        return 0
    # Zone E - left upper
    if act <= 70 and pred >= 180:
        return 8
    # Zone E - right lower
    if act >= 180 and pred <= 70:
        return 7
    # Zone D - right
    if act >= 240 and 70 <= pred <= 180:
        return 6
    # Zone D - left
    if act <= 70 <= pred <= 180:
        return 5
    # Zone C - upper
    if 70 <= act <= 290 and pred >= act + 110:
        return 4
    # Zone C - lower
    if 130 <= act <= 180 and pred <= (7/5) * act - 182:
        return 3
    # Zone B - upper
    if act < pred:
        return 2
    # Zone B - lower
    return 1

# Vectoritzar les funcions per l'eficiència
clarke_error_zone_detailed = np.vectorize(clarke_error_zone_detailed)

# Funció per calcular les zones d'accuracy
def zone_accuracy(act_arr, pred_arr, detailed=False, diabetes_type=1):
    """
    Calculates the average percentage of each zone based on Clarke
    Error Grid analysis for an array of predictions and an array of actual values
    """
    acc = np.zeros(9)
    res = clarke_error_zone_detailed(act_arr, pred_arr)
    acc_bin = np.bincount(res)
    acc[:len(acc_bin)] = acc_bin

    if not detailed:
        acc[1] = acc[1] + acc[2]
        acc[2] = acc[3] + acc[4]
        acc[3] = acc[5] + acc[6]
        acc[4] = acc[7] + acc[8]
        acc = acc[:5]

    return acc / sum(acc)

def plot_clarke_error(y_true, y_pred, filename="clarke_plot.png", title="Clarke Error Grid"):
    from sklearn.metrics import mean_absolute_error

    # Calcular zones detallades
    zones = clarke_error_zone_detailed(y_true, y_pred)

    # Definir colors i etiquetes per zona (detallades)
    zone_colors = ['green', 'orange', 'orange', 'red', 'red', 'purple', 'purple', 'black', 'black']
    zone_labels = ['A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E']

    plt.figure(figsize=(8, 8))

    # Pintar els punts segons la zona
    for z in range(9):
        idx = zones == z
        if np.any(idx):
            plt.scatter(y_true[idx], y_pred[idx], color=zone_colors[z], label=f"Zone {zone_labels[z]}", alpha=0.8, s=10)

    # Dibuixar línia ideal i ±20%
    x = np.linspace(0, 400, 1000)
    plt.plot(x, x, 'k--', linewidth=1, label="Ideal")
    plt.plot(x, 1.2 * x, 'g--', linewidth=0.8)
    plt.plot(x, 0.8 * x, 'g--', linewidth=0.8)

    # Línies de referència
    for val in [70, 180, 240, 290]:
        plt.axhline(val, color='gray', linestyle=':', linewidth=0.5)
        plt.axvline(val, color='gray', linestyle=':', linewidth=0.5)

    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.xlabel("Glucosa real (mg/dL)")
    plt.ylabel("Glucosa predita (mg/dL)")
    plt.title(title)
    plt.legend(loc="upper left", fontsize=10, title="Zones", title_fontsize='13', frameon=True, shadow=True, fancybox=True)
    plt.grid(False)
    plt.tight_layout()

    # Calcular mètriques
    rmse = glucose_rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score_custom(y_true, y_pred)

    # Accuracy de zones detallades
    acc = zone_accuracy(y_true, y_pred, detailed=True)

    # Combinar zones amb la mateixa lletra
    zone_summary = {
        'A': acc[0],
        'B': acc[1] + acc[2],
        'C': acc[3] + acc[4],
        'D': acc[5] + acc[6],
        'E': acc[7] + acc[8]
    }

    # Preparar text
    textstr = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}"
    for label, value in zone_summary.items():
        textstr += f"\nZona {label}: {value*100:.1f}%"

    # Mostrar el text sota el títol
    plt.text(
        0.2, 0.98,  # X molt proper a l'esquerra, Y a dalt
        textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='gray', boxstyle='round,pad=0.5')
    )


    # Guardar el gràfic
    plt.savefig(filename, dpi=300)
    plt.close()


