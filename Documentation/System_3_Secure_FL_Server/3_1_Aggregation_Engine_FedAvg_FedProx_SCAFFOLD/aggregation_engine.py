import os
import json
import numpy as np

def run_aggregation_simulation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parameter FL Server
    num_clients = 5
    rounds = 20
    mu_fedprox = 0.01
    
    print("=== [System 3.1] SIMULASI AGGREGATION ENGINE FL SERVER ===")
    print(f"Jumlah Klien (Welders): {num_clients}")
    print(f"Jumlah Ronde Komunikasi (T): {rounds}")
    print(f"Koefisien Proksimal FedProx (mu): {mu_fedprox}")

    # Cek berkas hasil eksperimen
    history_file = os.path.join(current_dir, "history_fedprox_nodp.json")
    results_file = os.path.join(current_dir, "fedprox_results.json")

    if not os.path.exists(history_file):
        history_file = os.path.join(r"c:\WAR-FL\Output\DeepLearning\history_fedprox_nodp.json")
        results_file = os.path.join(r"c:\WAR-FL\Output\DeepLearning\fedprox_results.json")

    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
        print(f"\nRiwayat Konvergensi Akurasi ({len(history)} Ronde):")
        for r, acc in enumerate(history, start=1):
            print(f"  Ronde {r:02d}/{rounds} -> Server Val Accuracy: {acc:.2f}%")

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            res = json.load(f)
        print("\nHasil Evaluasi Agregasi Akhir:")
        print(json.dumps(res, indent=4))

if __name__ == "__main__":
    run_aggregation_simulation()
