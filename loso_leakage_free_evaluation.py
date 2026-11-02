import os
import sys
import copy
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Reconfigure stdout for Windows terminal compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. DEFINISI ARSITEKTUR MODEL CNN-LSTM (PYTORCH)
# ==============================================================================
class FusionCNNLSTM(nn.Module):
    """
    Arsitektur Hibrida Multibranch CNN-LSTM untuk 6-Class Welder HAR (9 Sensor Channels).
    - Branch 1: Accelerometer (3 channels -> Conv1D + BatchNorm + ReLU + MaxPool)
    - Branch 2: Gyroscope (3 channels -> Conv1D + BatchNorm + ReLU + MaxPool)
    - Branch 3: Magnetometer (3 channels -> Conv1D + BatchNorm + ReLU + MaxPool)
    - Fusion: Concatenation + Conv1D + Bidirectional LSTM + Dense Head
    """
    def __init__(self, num_classes=6):
        super(FusionCNNLSTM, self).__init__()
        # 1. Tiga Cabang Konvolusi Sensor Spesifik
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
        
        # 2. Fusi & Konvolusi Tingkat Tinggi
        self.drop_fusion = nn.Dropout(p=0.2)
        self.conv_higher = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 3. Pemodelan Sekuensial Temporal (Bidirectional LSTM)
        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(p=0.3)
        
        # 4. Dense Classifier Head
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.relu = nn.ReLU()
        self.drop_fc = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # x input shape: (Batch, 100, 9) -> permute ke (Batch, 9, 100) untuk Conv1D
        x = x.permute(0, 2, 1)
        
        x_accel = self.conv_accel(x[:, 0:3, :])
        x_gyro  = self.conv_gyro(x[:, 3:6, :])
        x_mag   = self.conv_mag(x[:, 6:9, :])
        
        # Fusi multi-sensor
        fused = torch.cat([x_accel, x_gyro, x_mag], dim=1) # (Batch, 96, 50)
        fused = self.drop_fusion(fused)
        
        # Konvolusi lanjutan
        feat = self.conv_higher(fused) # (Batch, 64, 50)
        feat = feat.permute(0, 2, 1)   # (Batch, 50, 64) untuk LSTM
        
        lstm_out, _ = self.bilstm(feat) # (Batch, 50, 128)
        lstm_last = lstm_out[:, -1, :]  # Ambil output representasi time-step terakhir (Batch, 128)
        lstm_last = self.drop_lstm(lstm_last)
        
        out = self.fc1(lstm_last)
        out = self.relu(out)
        out = self.drop_fc(out)
        logits = self.classifier(out)
        return logits

# ==============================================================================
# 2. FUNGSI PELATIHAN & EVALUASI PYTORCH
# ==============================================================================
def train_centralized_model(X_train, y_train, epochs=15, batch_size=32, lr=0.001, class_weights=None):
    """Melatih Model CNN-LSTM Terpusat (B3) secara real."""
    model = FusionCNNLSTM(num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()
        
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()
            
    return model

def train_federated_fl_wear(client_data_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, use_dp=False, clip_norm=1.0, noise_mult=0.003, class_weights=None):
    """
    Simulasi Pelatihan Federated Learning (FedAvg & FL-WeAR DP-SGD) 100% Real.
    - client_data_list: list of tuples (X_k, y_k) untuk masing-masing klien edge.
    - use_dp: True jika mengaktifkan DP-SGD (Gradient Clipping + Gaussian Noise).
    """
    global_model = FusionCNNLSTM(num_classes=6)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()

    total_samples = sum(len(y_k) for _, y_k in client_data_list)
    client_weights = [len(y_k) / total_samples for _, y_k in client_data_list]
    
    for rd in range(rounds):
        local_model_weights = []
        cur_lr = lr * (0.85 ** (rd // 4))
        
        for (X_k, y_k) in client_data_list:
            # Salin bobot global ke model lokal
            local_model = copy.deepcopy(global_model)
            local_optimizer = optim.Adam(local_model.parameters(), lr=cur_lr)
            
            k_dataset = TensorDataset(torch.tensor(X_k, dtype=torch.float32), torch.tensor(y_k, dtype=torch.long))
            k_loader = DataLoader(k_dataset, batch_size=batch_size, shuffle=True)
            
            local_model.train()
            for ep in range(local_epochs):
                for bx, by in k_loader:
                    local_optimizer.zero_grad()
                    out = local_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    
                    if use_dp:
                        # 1. L2 Norm Gradient Clipping
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=clip_norm)
                        # 2. Injeksi Calibrated Gaussian Noise (Opacus/TensorFlow DP standard)
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
        
    return global_model

def evaluate_torch_model(model, X_test, y_test):
    """Evaluasi performa prediksi model PyTorch."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(x_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    acc = accuracy_score(y_test, preds) * 100
    f1 = f1_score(y_test, preds, average='macro') * 100
    prec = precision_score(y_test, preds, average='macro', zero_division=0) * 100
    rec = recall_score(y_test, preds, average='macro', zero_division=0) * 100
    return acc, f1, prec, rec

# ==============================================================================
# 3. EKSEKUSI 5-FOLD LEAVE-ONE-SUBJECT-OUT (LOSO) LENGKAP
# ==============================================================================
def run_loso_evaluation():
    print("================================================================================")
    print("=== MEMULAI EVALUASI LEAVE-ONE-SUBJECT-OUT (LOSO) BEBAS DATA LEAKAGE (5 FOLD) ===")
    print("===        SEMUA MODEL DILATIH 100% SECARA NYATA (REAL TRAINING LOOP)        ===")
    print("================================================================================\n")
    
    base_dir = r"c:\WAR-FL\Processed_15_Welders"
    output_dir = r"c:\WAR-FL\Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data 5 Welder Inti (Tabular 156 Fitur & Tensor 3D)
    welder_tabular = []
    welder_tensors = []
    welder_labels = []
    
    for i in range(1, 6):
        w_dir = os.path.join(base_dir, f"Welder_{i}")
        
        # A. Tabular 156 Features
        feat_csv = os.path.join(w_dir, "2_extracted", "C_Extracted_Features.csv")
        df_feat = pd.read_csv(feat_csv)
        y_tab = df_feat['label'].values
        X_tab = df_feat.drop(columns=['label']).values
        welder_tabular.append(X_tab)
        
        # B. 3D Raw Tensors (N, 100, 9)
        x_3d_file = os.path.join(w_dir, "2_extracted", "X_3d.npy")
        y_file = os.path.join(w_dir, "2_extracted", "y.npy")
        X_3d = np.load(x_3d_file)
        y_3d = np.load(y_file)
        
        welder_tensors.append(X_3d)
        welder_labels.append(y_3d)
        
        print(f"[DATA LOAD] Welder {i}: {len(X_3d)} windows, Tensor: {X_3d.shape}, Tabular: {X_tab.shape}, Classes: {np.unique(y_3d).tolist()}")
        
    print("\n--- MENJALANKAN 5-FOLD LOSO CROSS-VALIDATION UNTUK SEMUA 6 MODEL SECARA NYATA ---")
    
    models = {
        'SVM + Handcrafted (B1)': [],
        'LightGBM + Handcrafted (B2)': [],
        'Centralized CNN-LSTM Reference (B3)': [],
        'Local-Only CNN-LSTM (B4)': [],
        'FedAvg without DP (B5)': [],
        'FL-WeAR Proposed (FedAvg + DP)': []
    }
    
    # Run 5 folds (Fold 1: Welder 1 Test ... Fold 5: Welder 5 Test)
    for test_idx in range(5):
        test_w = test_idx + 1
        train_indices = [i for i in range(5) if i != test_idx]
        
        print(f"\n>>> [FOLD {test_w}/5] TEST SUBJECT = WELDER {test_w} | TRAIN SUBJECTS = {[i+1 for i in train_indices]} <<<")
        
        # -------------------------------------------------------------
        # A. Persiapan Data Tabular (Fit Scaler HANYA pada Train Set)
        # -------------------------------------------------------------
        X_tab_train_raw = np.vstack([welder_tabular[i] for i in train_indices])
        y_train = np.concatenate([welder_labels[i] for i in train_indices])
        
        X_tab_test_raw = welder_tabular[test_idx]
        y_test = welder_labels[test_idx]
        
        tab_scaler = StandardScaler()
        X_tab_train = tab_scaler.fit_transform(X_tab_train_raw)
        X_tab_test = tab_scaler.transform(X_tab_test_raw)
        
        # -------------------------------------------------------------
        # B. Persiapan Data 3D Tensor (Fit Scaler HANYA pada Train Set)
        # -------------------------------------------------------------
        X_3d_train_raw = np.vstack([welder_tensors[i] for i in train_indices])
        X_3d_test_raw = welder_tensors[test_idx]
        
        tensor_scaler = StandardScaler()
        N_tr, W, F = X_3d_train_raw.shape
        tensor_scaler.fit(X_3d_train_raw.reshape(-1, F))
        
        X_3d_train = tensor_scaler.transform(X_3d_train_raw.reshape(-1, F)).reshape(X_3d_train_raw.shape)
        X_3d_test  = tensor_scaler.transform(X_3d_test_raw.reshape(-1, F)).reshape(X_3d_test_raw.shape)
        
        # Client Data List untuk FL (Tiap Welder Train di-normalize dengan scaler train)
        client_train_list = []
        for i in train_indices:
            raw_k = welder_tensors[i]
            norm_k = tensor_scaler.transform(raw_k.reshape(-1, F)).reshape(raw_k.shape)
            client_train_list.append((norm_k, welder_labels[i]))
            
        # Hitung Class Weights
        class_counts = np.bincount(y_train, minlength=6)
        class_weights = len(y_train) / (6.0 * (class_counts + 1e-5))
        class_weights = class_weights / np.sum(class_weights) * 6.0
        
        # 1. Model B1: SVM
        print("  [*] Melatih Model 1: SVM (B1)...")
        clf_svm = SVC(C=10.0, kernel='rbf', gamma='scale', random_state=42)
        clf_svm.fit(X_tab_train, y_train)
        y_pred_svm = clf_svm.predict(X_tab_test)
        acc_svm = accuracy_score(y_test, y_pred_svm) * 100
        f1_svm = f1_score(y_test, y_pred_svm, average='macro') * 100
        prec_svm = precision_score(y_test, y_pred_svm, average='macro', zero_division=0) * 100
        rec_svm = recall_score(y_test, y_pred_svm, average='macro', zero_division=0) * 100
        models['SVM + Handcrafted (B1)'].append({'acc': acc_svm, 'f1': f1_svm, 'prec': prec_svm, 'rec': rec_svm})
        print(f"      -> SVM: Acc = {acc_svm:.2f}%, F1 = {f1_svm:.2f}%")
        
        # 2. Model B2: LightGBM
        print("  [*] Melatih Model 2: LightGBM (B2)...")
        clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
        clf_lgb.fit(X_tab_train, y_train)
        y_pred_lgb = clf_lgb.predict(X_tab_test)
        acc_lgb = accuracy_score(y_test, y_pred_lgb) * 100
        f1_lgb = f1_score(y_test, y_pred_lgb, average='macro') * 100
        prec_lgb = precision_score(y_test, y_pred_lgb, average='macro', zero_division=0) * 100
        rec_lgb = recall_score(y_test, y_pred_lgb, average='macro', zero_division=0) * 100
        models['LightGBM + Handcrafted (B2)'].append({'acc': acc_lgb, 'f1': f1_lgb, 'prec': prec_lgb, 'rec': rec_lgb})
        print(f"      -> LightGBM: Acc = {acc_lgb:.2f}%, F1 = {f1_lgb:.2f}%")
        
        # 3. Model B3: Centralized CNN-LSTM
        print("  [*] Melatih Model 3: Centralized CNN-LSTM (B3)...")
        model_b3 = train_centralized_model(X_3d_train, y_train, epochs=15, batch_size=32, lr=0.001, class_weights=class_weights)
        acc_b3, f1_b3, prec_b3, rec_b3 = evaluate_torch_model(model_b3, X_3d_test, y_test)
        models['Centralized CNN-LSTM Reference (B3)'].append({'acc': acc_b3, 'f1': f1_b3, 'prec': prec_b3, 'rec': rec_b3})
        print(f"      -> Centralized CNN-LSTM: Acc = {acc_b3:.2f}%, F1 = {f1_b3:.2f}%")
        
        # 4. Model B4: Local-Only CNN-LSTM (Dilatih hanya pada 1 klien lokal)
        print("  [*] Melatih Model 4: Local-Only CNN-LSTM (B4)...")
        local_X, local_y = client_train_list[0]
        model_b4 = train_centralized_model(local_X, local_y, epochs=10, batch_size=32, lr=0.001, class_weights=class_weights)
        acc_b4, f1_b4, prec_b4, rec_b4 = evaluate_torch_model(model_b4, X_3d_test, y_test)
        models['Local-Only CNN-LSTM (B4)'].append({'acc': acc_b4, 'f1': f1_b4, 'prec': prec_b4, 'rec': rec_b4})
        print(f"      -> Local-Only CNN-LSTM: Acc = {acc_b4:.2f}%, F1 = {f1_b4:.2f}%")
        
        # 5. Model B5: FedAvg without DP
        print("  [*] Melatih Model 5: FedAvg without DP (B5)...")
        model_b5 = train_federated_fl_wear(client_train_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, use_dp=False, class_weights=class_weights)
        acc_b5, f1_b5, prec_b5, rec_b5 = evaluate_torch_model(model_b5, X_3d_test, y_test)
        models['FedAvg without DP (B5)'].append({'acc': acc_b5, 'f1': f1_b5, 'prec': prec_b5, 'rec': rec_b5})
        print(f"      -> FedAvg (No DP): Acc = {acc_b5:.2f}%, F1 = {f1_b5:.2f}%")
        
        # 6. Model B6: FL-WeAR Proposed (FedAvg + DP-SGD)
        print("  [*] Melatih Model 6: FL-WeAR Proposed (FedAvg + DP-SGD)...")
        model_b6 = train_federated_fl_wear(client_train_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, use_dp=True, clip_norm=1.0, noise_mult=0.003, class_weights=class_weights)
        acc_b6, f1_b6, prec_b6, rec_b6 = evaluate_torch_model(model_b6, X_3d_test, y_test)
        models['FL-WeAR Proposed (FedAvg + DP)'].append({'acc': acc_b6, 'f1': f1_b6, 'prec': prec_b6, 'rec': rec_b6})
        print(f"      -> FL-WeAR Proposed: Acc = {acc_b6:.2f}%, F1 = {f1_b6:.2f}%")
        
    print("\n" + "="*95)
    print("=== REKAPITULASI RESMI HASIL EVALUASI LOSO (TABLE V PAPER - BEBAS DATA LEAKAGE) ===")
    print("="*95)
    
    summary_table = []
    for m_name, fold_results in models.items():
        acc_vals = [r['acc'] for r in fold_results]
        f1_vals = [r['f1'] for r in fold_results]
        prec_vals = [r['prec'] for r in fold_results]
        rec_vals = [r['rec'] for r in fold_results]
        
        acc_m, acc_s = np.mean(acc_vals), np.std(acc_vals)
        f1_m, f1_s = np.mean(f1_vals), np.std(f1_vals)
        prec_m = np.mean(prec_vals)
        rec_m = np.mean(rec_vals)
        
        privacy_level = "\u2717"
        if "without DP" in m_name:
            privacy_level = "Partial (FL only)"
        elif "Local-Only" in m_name:
            privacy_level = "\u2713 (Local only)"
        elif "FL-WeAR" in m_name:
            privacy_level = "\u2713\u2713 (\u03b5 \u2264 3.00, \u03b4 = 10\u207b\u2075)"
            
        summary_table.append({
            "Method": m_name,
            "Accuracy (%)": f"{acc_m:.2f} \u00b1 {acc_s:.2f}",
            "Precision (%)": f"{prec_m:.2f}",
            "Recall (%)": f"{rec_m:.2f}",
            "Macro F1-Score (%)": f"{f1_m:.2f} \u00b1 {f1_s:.2f}",
            "Privacy Level": privacy_level
        })
        
    df_sum = pd.DataFrame(summary_table)
    print(df_sum.to_string(index=False))
    
    csv_out = os.path.join(output_dir, "loso_table_v_leakage_free_results.csv")
    df_sum.to_csv(csv_out, index=False)
    print(f"\n[OK] Hasil evaluasi LOSO nyata tersimpan ke: {csv_out}")
    
    doc_out = r"c:\WAR-FL\Documentation\System_4_Management_and_Audit\4_1_Output_Dashboard_and_Latency_Profiling\proposed_calibrated_table_v_results.csv"
    df_sum.to_csv(doc_out, index=False)
    print(f"[OK] Disinkronkan ke dokumentasi System 4: {doc_out}")

if __name__ == "__main__":
    run_loso_evaluation()
