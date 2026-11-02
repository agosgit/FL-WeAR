# System 2.2: Local Training Process & Class-Weighted Loss

## 1. Deskripsi Komponen & Tujuan
Pada paradigma *Federated Learning* (FL), pelatihan model dilakukan secara mandiri di masing-masing perangkat klien (*worker node*) hanya menggunakan data lokal tanpa pernah mentransmisikan data mentah ke server. Karena aktivitas pengelasan di pabrik memiliki variasi durasi nyata yang tidak seimbang (misalnya pekerja lebih sering mengelas dalam posisi berdiri/berbaring dibandingkan posisi duduk), fungsi kerugian lokal dimodifikasi dengan **Class-Weighted Cross-Entropy Loss** agar model tidak bias terhadap kelas mayoritas.

---

## 2. File Input & Spesifikasi Data
* **Data Latih Lokal**: [`X_train_3d.npy`](file:///c:/WAR-FL/Documentation/System_2_Local_Client/2_2_Local_Training_and_Loss/X_train_3d.npy) $(2175, 100, 9)$.
* **Target Label**: [`y_train.npy`](file:///c:/WAR-FL/Documentation/System_2_Local_Client/2_2_Local_Training_and_Loss/y_train.npy) $(2175,)$.

---

## 3. Formulasi Matematis & Algoritma Pelatihan

### A. Class Weighting Factor
Untuk setiap kelas $c \in \{0, \dots, C-1\}$ dengan $C=6$:
$$w_c = \frac{N_{local}}{C \times N_{c}}$$
*di mana $N_{local}$ adalah total data pada klien dan $N_c$ adalah jumlah sampel kelas $c$.*

### B. Weighted Cross-Entropy Loss
$$\mathcal{L}_{local}(\theta) = - \frac{1}{B} \sum_{i=1}^B w_{y_i} \sum_{c=0}^{C-1} y_{i,c} \log \hat{y}_{i,c}(\theta)$$

### C. Local Hyperparameters
* **Local Epochs ($E$)**: $3 \dots 5$ epoch per ronde komunikasi FL.
* **Batch Size ($B$)**: 32 segmen per step optimasi.
* **Optimizer**: Adam ($\text{learning rate } \eta = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$).

---

## 4. Implementasi Kode Program
* **Skrip Mandiri**: [`local_training.py`](file:///c:/WAR-FL/Documentation/System_2_Local_Client/2_2_Local_Training_and_Loss/local_training.py)
* **Rujukan Notebook Utama**: [`main.ipynb`](file:///c:/WAR-FL/main.ipynb) pada **Cell 0, 1, 3, 5, 7, 9**.

```python
# Menghitung bobot kelas lokal
class_counts = np.bincount(y_train, minlength=6)
class_weights = len(y_train) / (6 * (class_counts + 1e-5))
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

# Pelatihan lokal per round
history = model.fit(
    X_train, y_train_cat,
    epochs=LOCAL_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    verbose=1
)
```

---

## 5. File Output yang Dihasilkan
* **Lokasi File Output**: [`local_training_metrics.json`](file:///c:/WAR-FL/Documentation/System_2_Local_Client/2_2_Local_Training_and_Loss/local_training_metrics.json)
  * Menyimpan log riwayat loss dan akurasi pelatihan lokal klien per epoch.

---

## 6. Korelasi dengan Naskah Paper (`paper_draft_filled.md`)
* **Section IV-B (Implementation Details, Line 115)**: Menjelaskan setup local epochs ($E=5$), batch size ($B=32$), learning rate ($\eta=0.001$), dan optimasi Adam.
* **TABLE III (FL-WeAR Hyperparameters)**: Rincian justifikasi pemilihan parameter pelatihan lokal edge device.
* **TABLE VIII (Ablation Study)**: Menunjukkan penurunan drastis Macro F1 sebesar **-57.23 pp** jika mekanisme *class balancing* dihilangkan.
