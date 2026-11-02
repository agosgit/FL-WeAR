import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Input, Dropout, Concatenate, Bidirectional

def build_fusion_cnn_lstm_model(input_shape=(100, 9), num_classes=6):
    """
    Arsitektur Hibrida Multibranch CNN-LSTM untuk 6-Class HAR IMU 9-Axis.
    """
    inputs = Input(shape=input_shape, name="input_layer")
    
    # 1. Pemisahan kanal sensor modalitas
    accel_input = inputs[:, :, 0:3]
    gyro_input  = inputs[:, :, 3:6]
    mag_input   = inputs[:, :, 6:9]

    # 2. Branch Konvolusi 1D Lokal
    x_accel = Conv1D(32, kernel_size=3, activation="relu", padding="same", name="conv_accel")(accel_input)
    x_accel = MaxPooling1D(pool_size=2, name="pool_accel")(x_accel)

    x_gyro = Conv1D(32, kernel_size=3, activation="relu", padding="same", name="conv_gyro")(gyro_input)
    x_gyro = MaxPooling1D(pool_size=2, name="pool_gyro")(x_gyro)

    x_mag = Conv1D(32, kernel_size=3, activation="relu", padding="same", name="conv_mag")(mag_input)
    x_mag = MaxPooling1D(pool_size=2, name="pool_mag")(x_mag)

    # 3. Fusi Fitur Multi-Sensor
    fused = Concatenate(axis=-1, name="sensor_fusion")([x_accel, x_gyro, x_mag])
    fused = Dropout(0.3, name="drop_fusion")(fused)

    # 4. Konvolusi Tingkat Tinggi & Pemodelan Temporal Sekuensial (LSTM)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same", name="conv_higher")(fused)
    x = Bidirectional(LSTM(64), name="bilstm_temporal")(x)
    x = Dropout(0.4, name="drop_lstm")(x)

    # 5. Kepala Klasifikasi (Dense Head)
    x = Dense(64, activation="relu", name="dense_classifier")(x)
    x = Dropout(0.3, name="drop_dense")(x)

    outputs = Dense(num_classes, activation="softmax", name="output_softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="FL_WeAR_CNN_LSTM_Fusion")
    return model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = build_fusion_cnn_lstm_model()
    
    print("=== MODEL ARCHITECTURE SUMMARY ===")
    model.summary()

    # Simpan ringkasan ke file teks
    summary_file = os.path.join(current_dir, "model_architecture_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"Summary tersimpan ke: {summary_file}")

if __name__ == "__main__":
    main()
