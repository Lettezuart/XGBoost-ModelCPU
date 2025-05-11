# main.py
import sys
import os
from tabulate import tabulate
import pandas as pd
from datetime import datetime
from models.offline_model import train_and_evaluate_offline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Afegeix el path del projecte per importar mòduls
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    base_path = "data/raw"
    results = train_and_evaluate_offline(base_path)

    patient_ids = sorted(results[30].keys())
    table = []

    # Afegir les mètriques de cada pacient
    for pid in patient_ids:
        row = [
            pid,
            results[30][pid]["rmse"],
            results[30][pid]["mse"],
            results[30][pid]["mae"],
            results[30][pid]["logloss"],
            results[30][pid]["r2"],
            results[60][pid]["rmse"],
            results[60][pid]["mae"],
            results[60][pid]["logloss"],
            results[60][pid]["r2"]
        ]
        table.append(row)

    # Última fila: promig
    avg_row = ["AVERAGE"]
    for horizon in [30, 60]:
        avg_rmse = sum(r["rmse"] for r in results[horizon].values()) / len(patient_ids)
        avg_mae = sum(r["mae"] for r in results[horizon].values()) / len(patient_ids)
        avg_logloss = sum(r["logloss"] for r in results[horizon].values()) / len(patient_ids)
        avg_r2 = sum(r["r2"] for r in results[horizon].values()) / len(patient_ids)
        
        avg_row.extend([avg_rmse, avg_mae, avg_logloss, avg_r2])
    table.append(avg_row)

    # Definir les capçaleres amb totes les mètriques
    headers = [
        "Pacient ID", 
        "RMSE 30", "MAE 30", "LogLoss 30", "R² 30",
        "RMSE 60", "MAE 60", "LogLoss 60", "R² 60"
    ]

    # Generar la taula com a string
    table_str = tabulate(table, headers=headers, tablefmt="grid")

    print("\n📊 Results table of the model Offline:")
    print(table_str)

    # Crear nom del fitxer amb datetime
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = "Taules_resultats"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"resultats_offline_{now}.txt")

    # Escriure la taula al fitxer
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("📊 Results table of the model Offline:\n")
        f.write(table_str)

    print(f"\n✅ Results saved at: {output_path}")


if __name__ == "__main__":
    main()
