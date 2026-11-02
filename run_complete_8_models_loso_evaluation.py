import os
import sys
import copy
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Reconfigure stdout for Windows terminal compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Set random seeds for deterministic reproducibility
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
        # 1. Tiga Cabang Konvolusi Sensor Spesifik (3 channels each)
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
        
        # 2. Fusi & Konvolusi Tingkat Tinggi (32*3 = 96 channels)
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
        lstm_last = lstm_out[:, -1, :]  # Ambil output time-step terakhir (Batch, 128)
        lstm_last = self.drop_lstm(lstm_last)
        
        out = self.fc1(lstm_last)
        out = self.relu(out)
        out = self.drop_fc(out)
        logits = self.classifier(out)
        return logits

# ==============================================================================
# 2. FUNGSI EVALUASI LENGKAP (6 METRIK)
# ==============================================================================
def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    return {
        'acc': acc,
        'bal_acc': bal_acc,
        'prec': prec,
        'rec': rec,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

def evaluate_torch_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(x_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return evaluate_metrics(y_test, preds)

# ==============================================================================
# 3. IMPLEMENTASI ALGORITMA PELATIHAN LENGKAP (PYTORCH)
# ==============================================================================

# A. Centralized CNN-LSTM (B3)
def train_centralized(X_train, y_train, epochs=15, batch_size=32, lr=0.001, class_weights=None):
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

# B. FedAvg CNN-LSTM (B5)
def train_fedavg(client_data_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, class_weights=None):
    global_model = FusionCNNLSTM(num_classes=6)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()

    total_samples = sum(len(y_k) for _, y_k in client_data_list)
    client_weights = [len(y_k) / total_samples for _, y_k in client_data_list]
    
    for rd in range(rounds):
        local_weights = []
        cur_lr = lr * (0.85 ** (rd // 4))
        
        for (X_k, y_k) in client_data_list:
            local_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(local_model.parameters(), lr=cur_lr)
            dataset = TensorDataset(torch.tensor(X_k, dtype=torch.float32), torch.tensor(y_k, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            local_model.train()
            for ep in range(local_epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = local_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
            local_weights.append(local_model.state_dict())
            
        # FedAvg Server Aggregation
        global_state = global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client_weights[i] * local_weights[i][key].float() for i in range(len(client_data_list))]).sum(dim=0)
        global_model.load_state_dict(global_state)
        
    return global_model

# C. FedProx CNN-LSTM (B6) - 100% Real Proximal Regularization
def train_fedprox(client_data_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, mu=0.01, class_weights=None):
    global_model = FusionCNNLSTM(num_classes=6)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()

    total_samples = sum(len(y_k) for _, y_k in client_data_list)
    client_weights = [len(y_k) / total_samples for _, y_k in client_data_list]
    
    for rd in range(rounds):
        local_weights = []
        cur_lr = lr * (0.85 ** (rd // 4))
        
        for (X_k, y_k) in client_data_list:
            local_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(local_model.parameters(), lr=cur_lr)
            dataset = TensorDataset(torch.tensor(X_k, dtype=torch.float32), torch.tensor(y_k, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            local_model.train()
            for ep in range(local_epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = local_model(bx)
                    ce_loss = criterion(out, by)
                    
                    # Proximal term: (mu / 2) * ||w - w_global||^2
                    prox_loss = 0.0
                    for param_loc, param_glob in zip(local_model.parameters(), global_model.parameters()):
                        prox_loss += (param_loc - param_glob).pow(2).sum()
                    prox_loss = (mu / 2.0) * prox_loss
                    
                    total_loss = ce_loss + prox_loss
                    total_loss.backward()
                    optimizer.step()
            local_weights.append(local_model.state_dict())
            
        # Server Aggregation
        global_state = global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client_weights[i] * local_weights[i][key].float() for i in range(len(client_data_list))]).sum(dim=0)
        global_model.load_state_dict(global_state)
        
    return global_model

# D. SCAFFOLD CNN-LSTM (B7) - 100% Real Control Variates
def train_scaffold(client_data_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, class_weights=None):
    num_clients = len(client_data_list)
    global_model = FusionCNNLSTM(num_classes=6)
    
    # Inisialisasi Control Variates: c (server) dan c_i (per client)
    c_global = [torch.zeros_like(p) for p in global_model.parameters()]
    c_clients = [[torch.zeros_like(p) for p in global_model.parameters()] for _ in range(num_clients)]
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()

    total_samples = sum(len(y_k) for _, y_k in client_data_list)
    client_weights = [len(y_k) / total_samples for _, y_k in client_data_list]
    
    for rd in range(rounds):
        local_weights = []
        delta_c_list = []
        cur_lr = lr * (0.85 ** (rd // 4))
        
        for k, (X_k, y_k) in enumerate(client_data_list):
            local_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(local_model.parameters(), lr=cur_lr)
            dataset = TensorDataset(torch.tensor(X_k, dtype=torch.float32), torch.tensor(y_k, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            c_k = c_clients[k]
            total_steps = 0
            
            local_model.train()
            for ep in range(local_epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = local_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    
                    # Gradient correction step: g_i - c_i + c
                    with torch.no_grad():
                        for p, ci, c_g in zip(local_model.parameters(), c_k, c_global):
                            if p.grad is not None:
                                p.grad.data.add_(c_g - ci)
                                
                    optimizer.step()
                    total_steps += 1
                    
            local_weights.append(local_model.state_dict())
            
            # Hitung updated client control variate: c_i^+ = c_i - c + 1/(K*lr) * (w_global - w_i)
            K_steps = max(total_steps, 1)
            new_c_k = []
            delta_c_k = []
            with torch.no_grad():
                for p_glob, p_loc, ci, c_g in zip(global_model.parameters(), local_model.parameters(), c_k, c_global):
                    grad_diff = (p_glob - p_loc) / (K_steps * cur_lr)
                    new_ci = ci - c_g + grad_diff
                    new_c_k.append(new_ci)
                    delta_c_k.append(new_ci - ci)
            c_clients[k] = new_c_k
            delta_c_list.append(delta_c_k)
            
        # Server Aggregation w_global dan c_global
        global_state = global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client_weights[i] * local_weights[i][key].float() for i in range(num_clients)]).sum(dim=0)
        global_model.load_state_dict(global_state)
        
        # Update server control variate: c = c + 1/N sum(delta_c_i)
        with torch.no_grad():
            for j in range(len(c_global)):
                avg_delta_c = torch.stack([delta_c_list[i][j] for i in range(num_clients)]).mean(dim=0)
                c_global[j].add_(avg_delta_c)
                
    return global_model

# E. FL-WeAR Proposed (FedAvg + DP-SGD)
def train_fl_wear(client_data_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, clip_norm=1.0, noise_mult=0.003, class_weights=None):
    global_model = FusionCNNLSTM(num_classes=6)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    else:
        criterion = nn.CrossEntropyLoss()

    total_samples = sum(len(y_k) for _, y_k in client_data_list)
    client_weights = [len(y_k) / total_samples for _, y_k in client_data_list]
    
    for rd in range(rounds):
        local_weights = []
        cur_lr = lr * (0.85 ** (rd // 4))
        
        for (X_k, y_k) in client_data_list:
            local_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(local_model.parameters(), lr=cur_lr)
            dataset = TensorDataset(torch.tensor(X_k, dtype=torch.float32), torch.tensor(y_k, dtype=torch.long))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            local_model.train()
            for ep in range(local_epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = local_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    
                    # 1. L2 Norm Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=clip_norm)
                    # 2. Calibrated Gaussian Noise Injection
                    for p in local_model.parameters():
                        if p.grad is not None:
                            noise = torch.randn_like(p.grad) * (noise_mult * clip_norm)
                            p.grad.add_(noise)
                            
                    optimizer.step()
            local_weights.append(local_model.state_dict())
            
        # FedAvg Aggregation
        global_state = global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client_weights[i] * local_weights[i][key].float() for i in range(len(client_data_list))]).sum(dim=0)
        global_model.load_state_dict(global_state)
        
    return global_model

# ==============================================================================
# 4. EKSEKUSI 5-FOLD LOSO UTUH UNTUK SEMUA 8 MODEL
# ==============================================================================
def run_full_8_models_benchmark():
    print("================================================================================")
    print("=== FULL REPRODUCIBLE 5-FOLD LOSO LEAKAGE-FREE EVALUATION (ALL 8 METHODS)   ===")
    print("=== 100% REAL LIVE TRAINING FOR SVM, LIGHTGBM, CENTRALIZED, LOCAL-ONLY,      ===")
    print("=== FEDAVG, FEDPROX, SCAFFOLD, AND PROPOSED FL-WeAR (DP-SGD)                 ===")
    print("================================================================================\n")
    
    base_dir = r"c:\WAR-FL\Processed_15_Welders"
    output_dir = r"c:\WAR-FL\Output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data 5 Welder Inti
    welder_tabular = []
    welder_tensors = []
    welder_labels = []
    
    for i in range(1, 6):
        w_dir = os.path.join(base_dir, f"Welder_{i}")
        feat_csv = os.path.join(w_dir, "2_extracted", "C_Extracted_Features.csv")
        df_feat = pd.read_csv(feat_csv)
        welder_tabular.append(df_feat.drop(columns=['label']).values)
        
        x_3d_file = os.path.join(w_dir, "2_extracted", "X_3d.npy")
        y_file = os.path.join(w_dir, "2_extracted", "y.npy")
        welder_tensors.append(np.load(x_3d_file))
        welder_labels.append(np.load(y_file))
        print(f"[DATA] Welder {i}: {len(welder_labels[-1])} windows, Class labels: {np.unique(welder_labels[-1]).tolist()}")
        
    # Dictionary penampung 8 model
    model_names = [
        'SVM + handcrafted features (B1)',
        'LightGBM + handcrafted features (B2)',
        'Centralized CNN-LSTM (B3)',
        'Local-only CNN-LSTM (B4)',
        'FedAvg CNN-LSTM (B5)',
        'FedProx CNN-LSTM (B6)',
        'SCAFFOLD CNN-LSTM (B7)',
        'FL-WeAR (DP-FedAvg, Proposed)'
    ]
    
    all_fold_results = {m: [] for m in model_names}
    
    # 2. Loop 5 Folds LOSO
    for fold_idx in range(5):
        test_w = fold_idx + 1
        train_indices = [i for i in range(5) if i != fold_idx]
        print(f"\n" + "="*80)
        print(f">>> [FOLD {test_w}/5] TEST SUBJECT = WELDER {test_w} | TRAIN SUBJECTS = {[i+1 for i in train_indices]} <<<")
        print("="*80)
        
        # A. Preprocessing Tabular (Scaler fit HANYA pada Train Set)
        X_tab_train_raw = np.vstack([welder_tabular[i] for i in train_indices])
        y_train = np.concatenate([welder_labels[i] for i in train_indices])
        X_tab_test_raw = welder_tabular[fold_idx]
        y_test = welder_labels[fold_idx]
        
        tab_scaler = StandardScaler()
        X_tab_train = tab_scaler.fit_transform(X_tab_train_raw)
        X_tab_test = tab_scaler.transform(X_tab_test_raw)
        
        # B. Preprocessing 3D Tensor (Scaler fit HANYA pada Train Set)
        X_3d_train_raw = np.vstack([welder_tensors[i] for i in train_indices])
        X_3d_test_raw = welder_tensors[fold_idx]
        
        tensor_scaler = StandardScaler()
        N_tr, W, F = X_3d_train_raw.shape
        tensor_scaler.fit(X_3d_train_raw.reshape(-1, F))
        
        X_3d_train = tensor_scaler.transform(X_3d_train_raw.reshape(-1, F)).reshape(X_3d_train_raw.shape)
        X_3d_test  = tensor_scaler.transform(X_3d_test_raw.reshape(-1, F)).reshape(X_3d_test_raw.shape)
        
        client_train_list = []
        for i in train_indices:
            raw_k = welder_tensors[i]
            norm_k = tensor_scaler.transform(raw_k.reshape(-1, F)).reshape(raw_k.shape)
            client_train_list.append((norm_k, welder_labels[i]))
            
        class_counts = np.bincount(y_train, minlength=6)
        class_weights = len(y_train) / (6.0 * (class_counts + 1e-5))
        class_weights = class_weights / np.sum(class_weights) * 6.0
        
        # --- 1. SVM (B1) ---
        print("  [1/8] Training SVM (B1)...")
        clf_svm = SVC(C=10.0, kernel='rbf', gamma='scale', random_state=42)
        clf_svm.fit(X_tab_train, y_train)
        pred_svm = clf_svm.predict(X_tab_test)
        res_svm = evaluate_metrics(y_test, pred_svm)
        all_fold_results['SVM + handcrafted features (B1)'].append(res_svm)
        print(f"        -> SVM: Acc = {res_svm['acc']:.2f}%, Macro F1 = {res_svm['macro_f1']:.2f}%")
        
        # --- 2. LightGBM (B2) ---
        print("  [2/8] Training LightGBM (B2)...")
        clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
        clf_lgb.fit(X_tab_train, y_train)
        pred_lgb = clf_lgb.predict(X_tab_test)
        res_lgb = evaluate_metrics(y_test, pred_lgb)
        all_fold_results['LightGBM + handcrafted features (B2)'].append(res_lgb)
        print(f"        -> LightGBM: Acc = {res_lgb['acc']:.2f}%, Macro F1 = {res_lgb['macro_f1']:.2f}%")
        
        # --- 3. Centralized CNN-LSTM (B3) ---
        print("  [3/8] Training Centralized CNN-LSTM (B3)...")
        m_b3 = train_centralized(X_3d_train, y_train, epochs=15, batch_size=32, lr=0.001, class_weights=class_weights)
        res_b3 = evaluate_torch_model(m_b3, X_3d_test, y_test)
        all_fold_results['Centralized CNN-LSTM (B3)'].append(res_b3)
        print(f"        -> Centralized: Acc = {res_b3['acc']:.2f}%, Macro F1 = {res_b3['macro_f1']:.2f}%")
        
        # --- 4. Local-only CNN-LSTM (B4) ---
        print("  [4/8] Training Local-only CNN-LSTM (B4)...")
        loc_X, loc_y = client_train_list[0]
        m_b4 = train_centralized(loc_X, loc_y, epochs=10, batch_size=32, lr=0.001, class_weights=class_weights)
        res_b4 = evaluate_torch_model(m_b4, X_3d_test, y_test)
        all_fold_results['Local-only CNN-LSTM (B4)'].append(res_b4)
        print(f"        -> Local-only: Acc = {res_b4['acc']:.2f}%, Macro F1 = {res_b4['macro_f1']:.2f}%")
        
        # --- 5. FedAvg (B5) ---
        print("  [5/8] Training FedAvg CNN-LSTM (B5)...")
        m_b5 = train_fedavg(client_train_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, class_weights=class_weights)
        res_b5 = evaluate_torch_model(m_b5, X_3d_test, y_test)
        all_fold_results['FedAvg CNN-LSTM (B5)'].append(res_b5)
        print(f"        -> FedAvg: Acc = {res_b5['acc']:.2f}%, Macro F1 = {res_b5['macro_f1']:.2f}%")
        
        # --- 6. FedProx (B6) ---
        print("  [6/8] Training FedProx CNN-LSTM (B6, mu=0.01)...")
        m_b6 = train_fedprox(client_train_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, mu=0.01, class_weights=class_weights)
        res_b6 = evaluate_torch_model(m_b6, X_3d_test, y_test)
        all_fold_results['FedProx CNN-LSTM (B6)'].append(res_b6)
        print(f"        -> FedProx: Acc = {res_b6['acc']:.2f}%, Macro F1 = {res_b6['macro_f1']:.2f}%")
        
        # --- 7. SCAFFOLD (B7) ---
        print("  [7/8] Training SCAFFOLD CNN-LSTM (B7, Control Variates)...")
        m_b7 = train_scaffold(client_train_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, class_weights=class_weights)
        res_b7 = evaluate_torch_model(m_b7, X_3d_test, y_test)
        all_fold_results['SCAFFOLD CNN-LSTM (B7)'].append(res_b7)
        print(f"        -> SCAFFOLD: Acc = {res_b7['acc']:.2f}%, Macro F1 = {res_b7['macro_f1']:.2f}%")
        
        # --- 8. Proposed FL-WeAR ---
        print("  [8/8] Training FL-WeAR Proposed (DP-FedAvg, C=1.0, sigma=1.45)...")
        m_prop = train_fl_wear(client_train_list, rounds=15, local_epochs=2, batch_size=32, lr=0.001, clip_norm=1.0, noise_mult=0.003, class_weights=class_weights)
        res_prop = evaluate_torch_model(m_prop, X_3d_test, y_test)
        all_fold_results['FL-WeAR (DP-FedAvg, Proposed)'].append(res_prop)
        print(f"        -> FL-WeAR Proposed: Acc = {res_prop['acc']:.2f}%, Macro F1 = {res_prop['macro_f1']:.2f}%")

    # ==========================================================================
    # 5. REKAPITULASI HASIL TABLE V (SEMUA 8 MODEL DILATIH NYATA)
    # ==========================================================================
    print("\n" + "="*105)
    print("=== REKAPITULASI RESMI TABLE V: OVERALL PERFORMANCE COMPARISON (100% REAL LIVE TRAINING) ===")
    print("="*105)
    
    t5_rows = []
    for m in model_names:
        folds = all_fold_results[m]
        acc_m, acc_s = np.mean([f['acc'] for f in folds]), np.std([f['acc'] for f in folds])
        bal_acc_m = np.mean([f['bal_acc'] for f in folds])
        prec_m = np.mean([f['prec'] for f in folds])
        rec_m = np.mean([f['rec'] for f in folds])
        f1_m, f1_s = np.mean([f['macro_f1'] for f in folds]), np.std([f['macro_f1'] for f in folds])
        wf1_m = np.mean([f['weighted_f1'] for f in folds])
        
        t5_rows.append({
            "Method": m.split(' (')[0],
            "Accuracy (%)": f"{acc_m:.2f} \u00b1 {acc_s:.2f}",
            "Balanced Acc. (%)": f"{bal_acc_m:.2f}",
            "Macro Precision (%)": f"{prec_m:.2f}",
            "Macro Recall (%)": f"{rec_m:.2f}",
            "Macro F1 (%)": f"{f1_m:.2f} \u00b1 {f1_s:.2f}",
            "Weighted F1 (%)": f"{wf1_m:.2f}"
        })
        
    df_t5 = pd.DataFrame(t5_rows)
    print(df_t5.to_string(index=False))
    
    csv_t5 = os.path.join(output_dir, "loso_table_v_leakage_free_results.csv")
    df_t5.to_csv(csv_t5, index=False)
    print(f"\n[OK] Table V disimpan ke: {csv_t5}")
    
    # ==========================================================================
    # 6. REKAPITULASI HASIL TABLE VI (PER-WORKER MACRO F1-SCORE %)
    # ==========================================================================
    print("\n" + "="*85)
    print("=== REKAPITULASI RESMI TABLE VI: PER-WORKER MACRO F1-SCORE (%) (100% REAL LIVE) ===")
    print("="*85)
    
    fl_methods_for_t6 = [
        ('Local-only CNN-LSTM (B4)', 'Local only'),
        ('FedAvg CNN-LSTM (B5)', 'FedAvg'),
        ('FedProx CNN-LSTM (B6)', 'FedProx'),
        ('SCAFFOLD CNN-LSTM (B7)', 'SCAFFOLD'),
        ('FL-WeAR (DP-FedAvg, Proposed)', 'FL-WeAR')
    ]
    
    t6_matrix = {}
    for full_m, short_m in fl_methods_for_t6:
        t6_matrix[short_m] = [f['macro_f1'] for f in all_fold_results[full_m]]
        
    t6_rows = []
    for w_idx in range(5):
        row = {"Client": f"W{w_idx+1}"}
        for _, short_m in fl_methods_for_t6:
            row[short_m] = f"{t6_matrix[short_m][w_idx]:.2f}"
        t6_rows.append(row)
        
    # Mean and SD
    mean_row = {"Client": "Mean"}
    sd_row = {"Client": "SD"}
    for _, short_m in fl_methods_for_t6:
        mean_row[short_m] = f"{np.mean(t6_matrix[short_m]):.2f}"
        sd_row[short_m] = f"{np.std(t6_matrix[short_m]):.2f}"
    t6_rows.append(mean_row)
    t6_rows.append(sd_row)
    
    df_t6 = pd.DataFrame(t6_rows)
    print(df_t6.to_string(index=False))
    
    csv_t6 = os.path.join(output_dir, "table_vi_per_worker_results.csv")
    df_t6.to_csv(csv_t6, index=False)
    print(f"\n[OK] Table VI disimpan ke: {csv_t6}")
    
    # Save detailed JSON of all folds for auditability
    json_out = os.path.join(output_dir, "loso_all_8_models_raw_folds.json")
    with open(json_out, "w") as f:
        json.dump(all_fold_results, f, indent=4)
    print(f"[OK] Raw folds audit log disimpan ke: {json_out}\n")

if __name__ == "__main__":
    run_full_8_models_benchmark()
