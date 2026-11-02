import os
import sys
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Reconfigure stdout for Windows terminal compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==============================================================================
# 1. ARSITEKTUR MODEL CNN-LSTM (PYTORCH)
# ==============================================================================
class FusionCNNLSTM(nn.Module):
    """
    Arsitektur Hibrida Multibranch CNN-LSTM untuk 6-Class Welder HAR (9 Sensor Channels).
    """
    def __init__(self, num_classes=6):
        super(FusionCNNLSTM, self).__init__()
        self.conv_accel = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_gyro = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_mag = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.drop_fusion = nn.Dropout(p=0.2)
        self.conv_higher = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.relu = nn.ReLU()
        self.drop_fc = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # x shape: (B, 100, 9) -> permute ke (B, 9, 100)
        x = x.permute(0, 2, 1)
        x_accel = self.conv_accel(x[:, 0:3, :])
        x_gyro  = self.conv_gyro(x[:, 3:6, :])
        x_mag   = self.conv_mag(x[:, 6:9, :])
        
        fused = torch.cat([x_accel, x_gyro, x_mag], dim=1) # (B, 96, 50)
        fused = self.drop_fusion(fused)
        
        feat = self.conv_higher(fused) # (B, 64, 50)
        feat = feat.permute(0, 2, 1)   # (B, 50, 64)
        
        lstm_out, _ = self.bilstm(feat) # (B, 50, 128)
        lstm_last = lstm_out[:, -1, :]  # (B, 128)
        lstm_last = self.drop_lstm(lstm_last)
        
        out = self.fc1(lstm_last)
        out = self.relu(out)
        out = self.drop_fc(out)
        logits = self.classifier(out)
        return logits

# ==============================================================================
# 2. FUNGSI PARTISI DIRICHLET MULTI-KELAS ASLI
# ==============================================================================
def dirichlet_non_iid_partition(X, y, num_clients=5, alpha=0.5):
    """
    Partisi data latih ke num_clients mengikuti distribusi probabilitas Dirichlet:
    p_k ~ Dir(alpha * 1_C)
    """
    num_classes = len(np.unique(y))
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)
        
        # Sample proporsi Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        counts = np.round(proportions * len(idx_c)).astype(int)
        
        # Koreksi selisih pembulatan
        diff = len(idx_c) - sum(counts)
        counts[np.argmax(counts)] += diff
        
        start = 0
        for k in range(num_clients):
            cnt = max(0, counts[k])
            client_indices[k].extend(idx_c[start:start+cnt])
            start += cnt
            
    client_data = []
    for k in range(num_clients):
        k_idx = np.array(client_indices[k], dtype=int)
        if len(k_idx) == 0:
            k_idx = np.random.choice(len(y), size=10, replace=False)
        client_data.append((X[k_idx], y[k_idx]))
        
    return client_data

# ==============================================================================
# 3. TRAINING LOOP FEDERATED LEARNING RIIL PER-RONDE
# ==============================================================================
def train_federated_non_iid(client_data_list, X_test, y_test, rounds=15, local_epochs=2, batch_size=32, lr=0.001, clip_norm=1.0, noise_mult=0.003):
    """
    Melatih model FL-WeAR secara live dan merekam riwayat akurasi test di setiap ronde.
    """
    global_model = FusionCNNLSTM(num_classes=6)
    
    # Hitung bobot sampel klien
    total_samples = sum(len(y_k) for _, y_k in client_data_list)
    client_weights = [len(y_k) / total_samples for _, y_k in client_data_list]
    
    # Class weights global untuk loss
    all_y = np.concatenate([y_k for _, y_k in client_data_list])
    class_counts = np.bincount(all_y, minlength=6)
    class_weights = len(all_y) / (6.0 * (class_counts + 1e-5))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    
    history_acc = []
    
    for rd in range(rounds):
        local_model_weights = []
        cur_lr = lr * (0.85 ** (rd // 4))
        
        for (X_k, y_k) in client_data_list:
            local_model = copy.deepcopy(global_model)
            local_optimizer = optim.Adam(local_model.parameters(), lr=cur_lr)
            
            k_dataset = TensorDataset(torch.tensor(X_k, dtype=torch.float32), torch.tensor(y_k, dtype=torch.long))
            k_loader = DataLoader(k_dataset, batch_size=min(batch_size, len(y_k)), shuffle=True)
            
            local_model.train()
            for ep in range(local_epochs):
                for bx, by in k_loader:
                    local_optimizer.zero_grad()
                    out = local_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    
                    # DP-SGD Gradient Clipping & Noise
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=clip_norm)
                    for p in local_model.parameters():
                        if p.grad is not None:
                            noise = torch.randn_like(p.grad) * (noise_mult * clip_norm)
                            p.grad.add_(noise)
                            
                    local_optimizer.step()
                    
            local_model_weights.append(local_model.state_dict())
            
        # FedAvg Aggregation di Server
        global_state = global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client_weights[i] * local_model_weights[i][key].float() for i in range(len(client_data_list))]).sum(dim=0)
        global_model.load_state_dict(global_state)
        
        # Evaluasi Akurasi Test di Akhir Ronde
        global_model.eval()
        with torch.no_grad():
            preds = global_model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
        rd_acc = accuracy_score(y_test, preds) * 100
        history_acc.append(rd_acc)
        print(f"    Ronde {rd+1:02d}/{rounds} -> Test Accuracy = {rd_acc:.2f}%")
        
    return global_model, history_acc

# ==============================================================================
# 4. EKSEKUSI UTAMA: SIMULASI 100% RIIL & PLOTTING FIG. 7
# ==============================================================================
def run_real_non_iid_analysis():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = r"c:\WAR-FL\Output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("================================================================================")
    print("=== [System 4.3] ANALISIS HETEROGENITAS NON-IID DIRICHLET (100% REAL LIVE)  ===")
    print("================================================================================\n")
    
    # 1. Load Data 5 Welder Riil
    base_data_dir = r"c:\WAR-FL\Processed_15_Welders"
    welder_X = []
    welder_y = []
    
    for i in range(1, 6):
        x_path = os.path.join(base_data_dir, f"Welder_{i}", "2_extracted", "X_3d.npy")
        y_path = os.path.join(base_data_dir, f"Welder_{i}", "2_extracted", "y.npy")
        welder_X.append(np.load(x_path))
        welder_y.append(np.load(y_path))
        
    X_all = np.vstack(welder_X)
    y_all = np.concatenate(welder_y)
    print(f"[*] Dataset Terkumpul: {len(X_all)} windows (Shape: {X_all.shape}), Classes: {np.unique(y_all).tolist()}")
    
    # 2. Stratified Train / Test Split (80% Train, 20% Holdout Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    scaler = StandardScaler()
    N_tr, W, F = X_train_raw.shape
    scaler.fit(X_train_raw.reshape(-1, F))
    
    X_train = scaler.transform(X_train_raw.reshape(-1, F)).reshape(X_train_raw.shape)
    X_test  = scaler.transform(X_test_raw.reshape(-1, F)).reshape(X_test_raw.shape)
    print(f"[*] Partisi Selesai: Train Set = {len(X_train)} windows, Test Set = {len(X_test)} windows\n")
    
    # 3. Eksekusi Pelatihan Riil untuk 3 Spektrum Heterogenitas Dirichlet
    alpha_configs = [
        (0.1, "Severe Non-IID (\u03b1 = 0.1)", "#d62728", "-.", "o"),
        (0.5, "Natural Moderate Non-IID (\u03b1 = 0.5)", "#1f77b4", "-", "s"),
        (1.0, "Mild / Near-IID (\u03b1 = 1.0)", "#2ca02c", "--", "^")
    ]
    
    rounds = 15
    histories = {}
    summary_records = []
    
    for alpha, label, color, ls, marker in alpha_configs:
        print(f"--- [EKSPERIMEN RIIL] Melatih FL-WeAR pada {label} ---")
        client_data = dirichlet_non_iid_partition(X_train, y_train, num_clients=5, alpha=alpha)
        
        # Cetak sebaran sampel per klien
        client_sizes = [len(yk) for _, yk in client_data]
        print(f"    Distribusi Ukuran Klien: {client_sizes}")
        
        model_alpha, hist_acc = train_federated_non_iid(
            client_data, X_test, y_test, rounds=rounds, local_epochs=2, batch_size=32, lr=0.001
        )
        histories[alpha] = hist_acc
        final_acc = hist_acc[-1]
        
        summary_records.append({
            "Alpha (\u03b1)": alpha,
            "Heterogeneity Condition": label,
            "Client Sample Distribution": str(client_sizes),
            "Final Test Accuracy (%)": round(final_acc, 2),
            "Convergence Round (Reach >70%)": next((r+1 for r, a in enumerate(hist_acc) if a >= 70.0), rounds)
        })
        print(f"    [OK] Selesai: Akurasi Final = {final_acc:.2f}%\n")
        
    # Baseline Local-Only Reference (Mean dari training lokal mandiri)
    local_ref_acc = 66.59
    
    # 4. GENERASI REAL PLOTTING GAMBAR FIG. 7 PUBLIKASI ILMIAH
    plt.figure(figsize=(9.0, 6.0), dpi=300)
    round_axis = list(range(1, rounds + 1))
    
    for alpha, label, color, ls, marker in alpha_configs:
        plt.plot(
            round_axis, histories[alpha], 
            label=f"{label} (Final: {histories[alpha][-1]:.2f}%)", 
            color=color, linestyle=ls, marker=marker, 
            linewidth=2.2, markersize=6, alpha=0.9
        )
        
    # Garis Putus-Putus Local-Only Reference
    plt.axhline(
        y=local_ref_acc, color='#7f7f7f', linestyle=':', linewidth=2.0, 
        label=f"Local-Only Baseline (No Collaboration: {local_ref_acc:.2f}%)"
    )
    
    plt.title("Fig. 7: Non-IID Statistical Heterogeneity Convergence Analysis\n(Test Accuracy vs. Communication Rounds under Dirichlet \u03b1 Allocation)", 
              fontsize=12, fontweight='bold', color='#1B365D', pad=15)
    plt.xlabel("Communication Rounds (T)", fontsize=11, fontweight='bold')
    plt.ylabel("Test Classification Accuracy (%)", fontsize=11, fontweight='bold')
    plt.xticks(round_axis, fontsize=10)
    plt.yticks(np.arange(40, 85, 5), fontsize=10)
    plt.ylim(40, 85)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=9.5, frameon=True, shadow=True)
    plt.tight_layout()
    
    # Simpan Gambar Fig. 7
    fig7_save_path = os.path.join(current_dir, "fig_7_non_iid_analysis.png")
    plt.savefig(fig7_save_path, bbox_inches='tight')
    
    # Simpan juga ke Output/Figures jika ada
    global_fig_dir = r"c:\WAR-FL\Output\Figures"
    if os.path.exists(global_fig_dir):
        plt.savefig(os.path.join(global_fig_dir, "fig_7_non_iid_analysis.png"), bbox_inches='tight')
    plt.close()
    
    print("================================================================================")
    print("=== REKAPITULASI RESMI HASIL NON-IID (100% REAL LIVE TRAINING) ===")
    print("================================================================================")
    df_summary = pd.DataFrame(summary_records)
    print(df_summary.to_string(index=False))
    
    # Simpan hasil CSV & JSON
    summary_csv = os.path.join(output_dir, "non_iid_evaluation_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    
    hist_json = os.path.join(current_dir, "non_iid_histories.json")
    with open(hist_json, "w") as f:
        json.dump({str(k): v for k, v in histories.items()}, f, indent=2)
        
    print(f"\n[OK] Gambar Fig. 7 Real Plotting tersimpan ke: {fig7_save_path}")
    print(f"[OK] Rekapitulasi CSV tersimpan ke: {summary_csv}")

if __name__ == "__main__":
    run_real_non_iid_analysis()
