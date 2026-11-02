# System 3.1: FL Server Aggregation Engine (FedAvg, FedProx, & SCAFFOLD)

## 1. Deskripsi Komponen & Tujuan
Server sentral pada arsitektur FL-WeAR berfungsi sebagai orkestrator pelatihan kolaboratif global. Server tidak memiliki akses ke data lokal pekerja las, melainkan hanya bertugas:
1. Menyiarkan bobot model global $w^t$ ke seluruh klien aktif pada awal setiap ronde komunikasi $t$.
2. Menerima pembaruan bobot terproteksi $\tilde{w}_k^{t+1}$ dari $K$ klien.
3. Melakukan agregasi parameter menggunakan algoritma terfederasi (**FedAvg**, **FedProx**, atau **SCAFFOLD**) untuk membentuk model global baru $w^{t+1}$.

---

## 2. Formulasi Matematis Algoritma Agregasi

### A. Federated Averaging (FedAvg)
Agregasi berbasis bobot proporsional terhadap ukuran data klien $n_k$:
$$w^{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_k^{t+1}, \quad \text{di mana } n = \sum_{k=1}^K n_k$$

### B. FedProx (Proximal Regularization)
Untuk mengatasi *client drift* akibat heterogenitas data non-IID antar-pekerja, FedProx menambahkan penalti jarak Euclidean terhadap model global pada loss lokal klien:
$$\mathcal{L}_{FedProx}(w; w^t) = \mathcal{L}_{local}(w) + \frac{\mu}{2} \|w - w^t\|_2^2$$
*dengan koefisien proksimal $\mu = 0.01$.*

### C. SCAFFOLD (Control Variates)
Menggunakan variabel kontrol klien $c_k$ dan global $c$ untuk mengoreksi bias gradien:
$$g_k^{corrected} = g_k - c_k + c$$
$$c_k^{t+1} = c_k^t - c^t + \frac{1}{K \eta_L} (w^t - w_k^{t+1})$$
$$c^{t+1} = \frac{1}{K} \sum_{k=1}^K c_k^{t+1}$$

---

## 3. Implementasi Kode Program
* **Skrip Mandiri**: [`aggregation_engine.py`](file:///c:/WAR-FL/Documentation/System_3_Secure_FL_Server/3_1_Aggregation_Engine_FedAvg_FedProx_SCAFFOLD/aggregation_engine.py)
* **Rujukan Notebook Utama**: [`main.ipynb`](file:///c:/WAR-FL/main.ipynb) pada **Cell 0–14, 33, 35, 37**.

```python
# Agregasi terbobot FedAvg di sisi Server
new_weights = []
for weights_list_tuple in zip(*client_weights):
    new_weights.append(np.average(weights_list_tuple, axis=0, weights=client_sizes))
global_weights = new_weights
```

---

## 4. File Output yang Dihasilkan
1. [`history_fedprox_nodp.json`](file:///c:/WAR-FL/Documentation/System_3_Secure_FL_Server/3_1_Aggregation_Engine_FedAvg_FedProx_SCAFFOLD/history_fedprox_nodp.json): Log akurasi validasi global per ronde komunikasi ($t = 1 \dots 20$).
2. [`fedprox_results.json`](file:///c:/WAR-FL/Documentation/System_3_Secure_FL_Server/3_1_Aggregation_Engine_FedAvg_FedProx_SCAFFOLD/fedprox_results.json): Ringkasan metrik akurasi ($79.34 \pm 0.91\%$) dan Macro F1 ($80.35 \pm 1.02\%$) 5-Fold CV.
3. [`fig_4_convergence_curves.png`](file:///c:/WAR-FL/Documentation/System_3_Secure_FL_Server/3_1_Aggregation_Engine_FedAvg_FedProx_SCAFFOLD/fig_4_convergence_curves.png): Kurva perbandingan konvergensi akurasi vs ronde komunikasi.

---

## 5. Korelasi dengan Naskah Paper (`paper_draft_filled.md`)
* **Section III (System Architecture, Line 35)**: Menguraikan fungsi System 3 dalam mengelola orkestrasi FL dan agregasi cloud.
* **TABLE V (Performance Comparison)**: Hasil komparasi performa FedAvg, FedProx, dan usulan FL-WeAR.
* **Section V-B (Federated Learning Convergence)**: Analisis kestabilan konvergensi dalam 20 ronde komunikasi.
