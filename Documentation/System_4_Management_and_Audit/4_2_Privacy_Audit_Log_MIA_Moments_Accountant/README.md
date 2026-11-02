# System 4.2: Privacy Audit Log (MIA Evaluation & Moments Accountant)

## 1. Deskripsi Komponen & Tujuan
Untuk memenuhi regulasi perlindungan data pekerja (GDPR dan UU PDP Indonesia), sistem FL-WeAR dilengkapi dengan modul audit privasi independen. Modul ini bertugas:
1. **Mengukur Ketahanan terhadap Serangan Inferensi Keanggotaan (*Membership Inference Attack* - MIA)**: Menguji apakah penyerang (*adversary*) dapat menebak dengan benar apakah data sensor pekerja tertentu digunakan dalam pelatihan model global.
2. **Akuntansi Pengeluaran Budget Privasi (*Moments Accountant*)**: Menghitung akumulasi komposisi budget privasi formal $\varepsilon_{total}$ pada tingkat kepercayaan $\delta = 10^{-5}$ setelah $T=20$ ronde federasi.

---

## 2. Formulasi Matematis Audit Privasi

### A. Membership Inference Attack (MIA)
Diberikan model terlatih $f_\theta$ dan sampel $(x, y)$:
$$\text{MIA}(x, y) = \mathbb{I}\left( \max_{c} f_\theta(x)_c \ge \tau \right)$$
*Jika akurasi serangan mendekati **50.00%** (setara pelemparan koin acak), maka model memiliki ketahanan privasi teoritis optimal.*

### B. Rényi Differential Privacy (RDP) Moments Accountant
Akumulasi konsumsi budget privasi setelah $T$ langkah sampling sub-klien:
$$\varepsilon_{total}(\delta) = \min_{\alpha > 1} \left\{ T \cdot \epsilon_{RDP}(\alpha) + \frac{\log(1/\delta)}{\alpha - 1} \right\}$$

---

## 3. Hasil Audit Privasi (Table VII Paper)

| Metode / Konfigurasi | Target $\varepsilon$ | $\delta$ | MIA Accuracy (%) | $\varepsilon_{total}$ (setelah 20 rds) | Tingkat Proteksi Privasi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Centralized CNN-LSTM (B3)** | $\infty$ | — | 62.40% | N/A | **Nol** (Data mentah terekspos) |
| **FL FedAvg, no DP (B5)** | $\infty$ | — | 57.64% | N/A | **Parsial** (Hanya arsitektur) |
| **FL + DP $\varepsilon=10$** | 10.0 | $10^{-5}$ | 53.58% | 19.0 | **Rendah** |
| **FL-WeAR (Proposed $\varepsilon=3$)** | **3.0** | **$10^{-5}$** | **51.27%** | **5.7** | **Kuat (Optimal Knee Point)** |
| **FL + DP $\varepsilon=1$** | 1.0 | $10^{-5}$ | 50.23% | 1.9 | **Sangat Kuat** |

---

## 4. Implementasi Kode Program
* **Skrip Mandiri**: [`privacy_audit_mia.py`](file:///c:/WAR-FL/Documentation/System_4_Management_and_Audit/4_2_Privacy_Audit_Log_MIA_Moments_Accountant/privacy_audit_mia.py)
* **Rujukan Notebook Utama**: [`main.ipynb`](file:///c:/WAR-FL/main.ipynb) pada **Cell 38–40**.

---

## 5. File Output yang Dihasilkan
1. [`table_vi_privacy_results.csv`](file:///c:/WAR-FL/Documentation/System_4_Management_and_Audit/4_2_Privacy_Audit_Log_MIA_Moments_Accountant/table_vi_privacy_results.csv): Rekapitulasi kuantitatif audit privasi MIA dan RDP.
2. [`fig_6_privacy_utility_tradeoff.png`](file:///c:/WAR-FL/Documentation/System_4_Management_and_Audit/4_2_Privacy_Audit_Log_MIA_Moments_Accountant/fig_6_privacy_utility_tradeoff.png): Kurva tradeoff privasi-utilitas ($\varepsilon$ vs Akurasi) yang mengonfirmasi knee-point pada $\varepsilon = 3.0$.

---

## 6. Korelasi dengan Naskah Paper (`paper_draft_filled.md`)
* **Section V-D (Privacy Analysis, Line 203)**: Pembahasan penurunan akurasi MIA ke 51.27% dan total privacy budget $\varepsilon_{total} = 5.7$.
* **TABLE VII (Privacy Evaluation Results)**: Tabel resmi hasil evaluasi MIA dan Moments Accountant.
* **Fig. 6 (Privacy-Utility Tradeoff Curve)**: Visualisasi titik optimal kompromi privasi-akurasi.
