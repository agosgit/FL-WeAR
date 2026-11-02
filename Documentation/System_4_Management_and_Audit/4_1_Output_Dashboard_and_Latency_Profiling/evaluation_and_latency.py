import os
import sys
import pandas as pd

# Reconfigure encoding for Windows stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_evaluation_summary():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Metrik Hasil Evaluasi Per-Kelas Table V / VI
    calibrated_results_file = os.path.join(current_dir, "proposed_calibrated_table_v_results.csv")
    latency_file = os.path.join(current_dir, "proposed_latency_summary.csv")
    cv_file = os.path.join(current_dir, "5_fold_cv_results.csv")

    if not os.path.exists(calibrated_results_file):
        calibrated_results_file = os.path.join(r"c:\WAR-FL\Output\DeepLearning\proposed_calibrated_table_v_results.csv")
        latency_file = os.path.join(r"c:\WAR-FL\Output\CNN_LSTM_FUSION\proposed_latency_summary.csv")

    print("=== [System 4.1] EVALUASI PERFORMA AKTIVITAS PENGELASAN & PROFILING LATENSI ===")
    
    if os.path.exists(cv_file):
        df_cv = pd.read_csv(cv_file)
        print("\n1. Stratified 5-Fold Cross-Validation Performance Breakdown:")
        print(df_cv.to_string(index=False))

    if os.path.exists(calibrated_results_file):
        df_cal = pd.read_csv(calibrated_results_file)
        print("\n2. Per-Class Recognition Performance (Table VI Paper):")
        print(df_cal.to_string(index=False))

    if os.path.exists(latency_file):
        df_lat = pd.read_csv(latency_file)
        print("\n3. Real-Time Pipeline Latency Breakdown (ms):")
        print(df_lat.to_string(index=False))

if __name__ == "__main__":
    run_evaluation_summary()
