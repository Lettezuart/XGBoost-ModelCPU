# main.py
import sys
import os
from tabulate import tabulate
import pandas as pd
from datetime import datetime
from models.offline_model import train_and_evaluate_offline

# Mostra totes les columnes i amplia l'ample per consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Afegeix el path del projecte per importar mòduls
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    base_path = "data/raw"
    results = train_and_evaluate_offline(base_path)

    patient_ids = sorted(results[30].keys())
    table = []

    for pid in patient_ids:
        row = [
            pid,
            results[30][pid]["rmse"],
            results[30][pid]["mae"],
            results[60][pid]["rmse"],
            results[60][pid]["mae"]
        ]
        table.append(row)

    # Última fila: promig
    avg_row = ["PROMIG"]
    for horizon in [30, 60]:
        avg_rmse = sum(r["rmse"] for r in results[horizon].values()) / len(patient_ids)
        avg_mae = sum(r["mae"] for r in results[horizon].values()) / len(patient_ids)
        avg_row.extend([avg_rmse, avg_mae])
    table.append(avg_row)

    headers = ["Pacient ID", "RMSE 30", "MAE 30", "RMSE 60", "MAE 60"]

    # Genera taula com string
    table_str = tabulate(table, headers=headers, tablefmt="grid")

    print("\n📊 Taula resum de resultats model Offline:")
    print(table_str)

    # Crear nom del fitxer amb datetime
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = "ResultatsTemporals/taules_resultats"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"resultats_offline_{now}.txt")

    # Escriure taula al fitxer
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("📊 Taula resum de resultats model Offline:\n")
        f.write(table_str)

    print(f"\n✅ Resultats guardats a: {output_path}")

if __name__ == "__main__":
    main()
