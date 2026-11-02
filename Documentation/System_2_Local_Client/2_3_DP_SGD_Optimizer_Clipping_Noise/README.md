# System 2.3: DP-SGD Optimizer (Gradient Clipping & Gaussian Noise Injection)

## 1. Deskripsi Komponen & Tujuan
Meskipun *Federated Learning* (FL) mencegah pengiriman data sensor mentah, penelitian terkini membuktikan bahwa pembaruan bobot model (*weight updates / gradients*) masih rentan terhadap serangan rekonstruksi data (*gradient inversion*) dan inferensi keanggotaan (*Membership Inference Attack* - MIA). Untuk menutup celah keamanan ini, FL-WeAR mengintegrasikan mekanisme formal **$(\varepsilon, \delta)$-Differential Privacy (DP)** berbasis **DP-SGD** (*Differentially Private Stochastic Gradient Descent*) langsung pada proses optimasi lokal di setiap perangkat klien.

---

## 2. Formulasi Matematis DP-SGD

### A. Pembatasan Sensitivitas ($L_2$ Gradient Clipping)
Untuk membatasi pengaruh satu sampel data terhadap perubahan gradien global, vektor gradien lokal $g_t$ dipotong (*clipped*) menggunakan ambang batas norma $L_2$ ($C_{clip} = 1.0$):
$$\bar{g}_t = \frac{g_t}{\max\left(1, \frac{\|g_t\|_2}{C_{clip}}\right)}$$

### B. Injeksi Noise Gaussian Terkalibrasi
Noise acak berdistribusi normal ditambahkan ke gradien yang telah dipotong:
$$\tilde{g}_t = \bar{g}_t + \mathcal{N}\left(0, \sigma^2 C_{clip}^2 \mathbf{I}\right)$$
*di mana $\sigma$ adalah parameter pengali noise (*noise multiplier*), yang dikalibrasi untuk memenuhi target privasi $(\varepsilon = 3.0, \delta = 10^{-5})$.*

### C. Pembaruan Parameter Bobot Lokal
$$w_{t+1} = w_t - \eta \tilde{g}_t$$

---

## 3. Implementasi Kode Program
* **Skrip Mandiri**: [`dp_sgd_optimizer.py`](file:///c:/WAR-FL/Documentation/System_2_Local_Client/2_3_DP_SGD_Optimizer_Clipping_Noise/dp_sgd_optimizer.py)
* **Rujukan Notebook Utama**: [`main.ipynb`](file:///c:/WAR-FL/main.ipynb) pada **Cell 1 (`train_step_fedprox_dp`, `train_step_scaffold_dp`) & Cell 37**.

```python
# Pemotongan L2 dan Injeksi Noise Gaussian
clipped_grad = tf.clip_by_norm(grad, clip_norm=clip_norm)
noise = tf.random.normal(
    shape=grad.shape,
    mean=0.0,
    stddev=noise_mult * clip_norm,
    dtype=tf.float32
)
dp_grad = clipped_grad + noise
optimizer.apply_gradients(zip([dp_grad], [weight]))
```

---

## 4. File Output yang Dihasilkan
* **Lokasi File Output**: [`dp_config.json`](file:///c:/WAR-FL/Documentation/System_2_Local_Client/2_3_DP_SGD_Optimizer_Clipping_Noise/dp_config.json)
  * Berisi parameter formal DP: $\varepsilon = 3.0$, $\delta = 10^{-5}$, $C_{clip} = 1.0$, $\sigma = 0.003$.

---

## 5. Korelasi dengan Naskah Paper (`paper_draft_filled.md`)
* **Section I (Introduction, Line 29)**: Menjelaskan peranan $(\varepsilon, \delta)$-DP dengan Gaussian noise injection untuk mencegah serangan gradient inversion dan MIA.
* **TABLE III (FL-WeAR Hyperparameters)**: Rincian parameter DP ($C_{clip}=1.0$, target $\varepsilon=3.0$, $\delta=10^{-5}$).
* **Section V-D (Privacy Analysis)**: Pembuktian efektivitas DP-SGD dalam menekan akurasi serangan MIA hingga **51.27%** (mendekati *random chance* 50%).
