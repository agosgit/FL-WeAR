import os
import sys
import json
import numpy as np

# Reconfigure encoding for Windows stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score

# Import model architecture builder from 2.1
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "2_1_CNN_LSTM_Architecture")))
from model_architecture import build_fusion_cnn_lstm_model

def run_local_training():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load input data
    x_train_path = os.path.join(current_dir, "X_train_3d.npy")
    y_train_path = os.path.join(current_dir, "y_train.npy")

    if not os.path.exists(x_train_path):
        x_train_path = os.path.join(r"c:\WAR-FL\Output\Segmentation\X_train_3d.npy")
        y_train_path = os.path.join(r"c:\WAR-FL\Output\Segmentation\y_train.npy")

    print(f"[System 2.2] Memuat data lokal client dari: {x_train_path}")
    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    
    num_classes = 6
    batch_size = 32
    local_epochs = 3
    lr = 0.001

    # 1. Hitung bobot kelas (Class Weights) untuk mengatasi ketimpangan data
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_weights = len(y_train) / (num_classes * (class_counts + 1e-5))
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print(f"Distribusi Sampel Kelas: {class_counts}")
    print(f"Bobot Kelas (Class Weights): {class_weight_dict}")

    # 2. Inisialisasi Model Lokal
    model = build_fusion_cnn_lstm_model(input_shape=(100, 9), num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    y_train_cat = to_categorical(y_train, num_classes)

    # 3. Pelatihan Lokal
    print(f"\n[System 2.2] Memulai pelatihan lokal selama {local_epochs} epoch (Batch Size={batch_size})...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=local_epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        verbose=2
    )

    # Evaluasi Prediksi Lokal
    y_pred_prob = model.predict(X_train, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = float(accuracy_score(y_train, y_pred) * 100)
    f1 = float(f1_score(y_train, y_pred, average="macro") * 100)
    
    metrics = {
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_accuracy": round(acc, 2),
        "train_macro_f1": round(f1, 2),
        "loss_history": [float(l) for l in history.history["loss"]],
        "acc_history": [float(a) for a in history.history["accuracy"]]
    }

    metrics_file = os.path.join(current_dir, "local_training_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n[System 2.2] Hasil Pelatihan Lokal: Acc = {acc:.2f}%, F1 = {f1:.2f}%")
    print(f"Metrik tersimpan ke: {metrics_file}")

if __name__ == "__main__":
    run_local_training()
