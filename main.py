# main.py
import sys
import os
from tabulate import tabulate
import pandas as pd
from datetime import datetime
from models.offline_model import train_and_evaluate_offline
from utils.final_validation import final_validation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path so imports resolve
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    base_path = "data/raw"

    # 1) Offline training & evaluation
    offline_results = train_and_evaluate_offline(base_path)

    patient_ids = sorted(offline_results[30].keys())
    offline_table = []

    # Build offline results table for horizons 30 and 60
    # Obtenim la unió de pacients dels dos horitzons
    patient_ids = sorted(set(offline_results[30].keys()).union(offline_results[60].keys()))
    offline_table = []

    # Build offline results table for horizons 30 and 60
    for pid in patient_ids:
        r_30 = offline_results[30].get(pid)
        r_60 = offline_results[60].get(pid)

        row = [pid]

        if r_30:
            row.extend([r_30["rmse"], r_30["mae"], r_30["r2"]])
        else:
            print(f"⚠️ No metrics for patient {pid} at 30 min")
            row.extend(["-", "-", "-"])

        if r_60:
            row.extend([r_60["rmse"], r_60["mae"], r_60["r2"]])
        else:
            print(f"⚠️ No metrics for patient {pid} at 60 min")
            row.extend(["-", "-", "-"])

        offline_table.append(row)


    avg_row = ["AVERAGE"]
    for horizon in [30, 60]:
        avg_rmse = sum(r["rmse"] for r in offline_results[horizon].values()) / len(patient_ids)
        avg_mae  = sum(r["mae"] for r in offline_results[horizon].values()) / len(patient_ids)
        avg_r2   = sum(r["r2"]  for r in offline_results[horizon].values()) / len(patient_ids)
        avg_row.extend([avg_rmse, avg_mae, avg_r2])
    offline_table.append(avg_row)

    offline_headers = [
        "Patient ID",
        "RMSE 30", "MAE 30", "R² 30",
        "RMSE 60", "MAE 60", "R² 60"
    ]

    offline_str = tabulate(offline_table, headers=offline_headers, tablefmt="grid")
    print("\n📊 Offline results:")
    print(offline_str)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir_training = "Training_metrics"
    os.makedirs(output_dir_training, exist_ok=True)
    offline_path = os.path.join(output_dir_training, f"resultats_offline_{now}.txt")
    with open(offline_path, "w", encoding="utf-8") as f:
        f.write("📊 Offline results:\n")
        f.write(offline_str)
    print(f"✅ Saved offline table to {offline_path}")

    # 2) Final validation on held-out test sets
    print("\n🔬 Running final validation...")
    output_dir_validation = "Validation_metrics"
    os.makedirs(output_dir_validation, exist_ok=True)
    final_results = final_validation(base_path)

    for horizon in [30, 60]:
        final_table = []
        patient_ids = sorted(set().union(*[r.keys() for r in final_results.values()]))

        for pid in patient_ids:
            metrics = final_results[horizon].get(pid)
            if metrics is None:
                print(f"⚠️ No metrics found for patient {pid} at horizon {horizon}")
                continue
            final_table.append([pid, metrics["rmse"], metrics["mae"], metrics["r2"]])

        if not final_table:
            print(f"❌ No results available for horizon {horizon}. Skipping table generation.")
            continue

        # Average metrics
        avg_rmse = sum(row[1] for row in final_table) / len(final_table)
        avg_mae  = sum(row[2] for row in final_table) / len(final_table)
        avg_r2   = sum(row[3] for row in final_table) / len(final_table)
        final_table.append(["AVERAGE", avg_rmse, avg_mae, avg_r2])

        # Print and save
        final_headers = ["Patient ID", "RMSE", "MAE", "R²"]
        final_str = tabulate(final_table, headers=final_headers, tablefmt="grid")
        print(f"\n📋 Final validation ({horizon}min):")
        print(final_str)

        final_path = os.path.join(output_dir_validation, f"final_validation_h{horizon}min_{now}.txt")
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(f"📋 Final validation ({horizon}min):\n")
            f.write(final_str)
        print(f"✅ Saved final validation table to {final_path}")


if __name__ == "__main__":
    main()
