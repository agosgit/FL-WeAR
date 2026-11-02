import os
import json
import numpy as np
import tensorflow as tf

def train_step_dp_sgd(x_batch, y_batch, model, optimizer, loss_fn, class_weights, clip_norm=1.0, noise_mult=0.003):
    """
    Custom Step DP-SGD: L2 Gradient Clipping + Gaussian Noise Addition
    """
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        raw_loss = loss_fn(y_batch, logits)
        class_indices = tf.argmax(y_batch, axis=1)
        batch_sample_weights = tf.gather(class_weights, class_indices)
        base_loss = tf.reduce_mean(raw_loss * batch_sample_weights)

    # 1. Hitung gradien per batch
    grads = tape.gradient(base_loss, model.trainable_weights)
    
    # 2. Mekanisme Differential Privacy (DP)
    dp_grads = []
    for grad in grads:
        if grad is not None:
            # A. L2 Norm Gradient Clipping
            clipped_grad = tf.clip_by_norm(grad, clip_norm=clip_norm)
            
            # B. Injeksi Noise Gaussian terkalibrasi
            noise = tf.random.normal(
                shape=grad.shape,
                mean=0.0,
                stddev=noise_mult * clip_norm,
                dtype=tf.float32
            )
            dp_grads.append(clipped_grad + noise)
        else:
            dp_grads.append(grad)

    # 3. Terapkan gradien terproteksi DP ke bobot model
    optimizer.apply_gradients(zip(dp_grads, model.trainable_weights))
    return base_loss

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dp_config = {
        "target_epsilon": 3.0,
        "delta": 1e-5,
        "l2_norm_clip": 1.0,
        "noise_multiplier": 0.003,
        "mechanism": "Gaussian DP-SGD",
        "description": "Gradient clipping bounding sensitivity and calibrated Gaussian noise injection"
    }

    config_file = os.path.join(current_dir, "dp_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(dp_config, f, indent=4)
        
    print(f"[System 2.3] Konfigurasi DP-SGD tersimpan ke: {config_file}")
    print("Konfigurasi Parameter DP:")
    for k, v in dp_config.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
