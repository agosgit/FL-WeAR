# System 4.3: Worker Monitor & Non-IID Heterogeneity Analysis

## 1. Deskripsi Komponen & Tujuan
Setiap pekerja las memiliki gaya kerja, kecepatan gerakan, serta proporsi waktu pengelasan yang unik. Karakteristik ini menyebabkan distribusi data sensor antar-pekerja bersifat sangat heterogen (*Non-Identical and Independently Distributed* / Non-IID). Modul Worker Monitor bertugas:
1. **Memantau Dinamika Pembelajaran Antar-Klien**: Memeriksa persebaran akurasi lokal dan mengukur *client drift* selama ronde agregasi berlangsung.
2. **Kuantifikasi Dampak Heterogenitas Statistik**: Mensimulasikan alokasi Dirichlet ($\alpha$) untuk membuktikan ketangguhan FL-WeAR di berbagai skenario variasi data pekerja.

---

## 2. Pemodelan Distribusi Non-IID Dirichlet ($\alpha$)
Pembagian data lokal tiap klien $k$ disintesis mengikuti distribusi probabilitas Dirichlet multi-kelas:
$$p_k \sim \text{Dir}(\alpha \cdot \mathbf{1}_C)$$
* **$\alpha = 0.1$ (Severe Non-IID)**: Heterogenitas ekstrem, di mana setiap pekerja hanya melakukan segelintir postur tertentu.
* **$\alpha = 0.5$ (Natural Non-IID - Default Setting)**: Kondisi alami heterogenitas bengkel las aktual.
* **$\alpha = 1.0$ (Near-IID)**: Distribusi kelas mendekati seragam antar-pekerja.

---

## 3. Hasil Analisis Konvergensi & Ketahanan Non-IID (100% Real Live Training)

| Nilai $\alpha$ | Tingkat Heterogenitas | Sebaran Sampel Klien | Akurasi Final (%) | Ronde Capai >70% |
| :---: | :--- | :---: | :---: | :---: |
| **0.1** | **Severe Non-IID** | $[470, 487, 473, 534, 208]$ | **55.62%** | $15$ (Fluktuasi gradien ekstrem) |
| **0.5** | **Natural Moderate Non-IID** | $[277, 678, 451, 206, 560]$ | **71.45%** | **Ronde 12** |
| **1.0** | **Mild / Near-IID** | $[457, 236, 452, 446, 581]$ | **75.32%** | **Ronde 10** |

### Temuan Utama (Live Training):
1. **Dampak Heterogenitas Statistik**: Nilai $\alpha = 1.0$ konvergen paling cepat dan mencapai **$75.32\%$**, disusul oleh $\alpha = 0.5$ pada **$71.45\%$**.
2. **Ketahanan Ekstrem ($\alpha = 0.1$)**: Pada $\alpha = 0.1$, disparitas kelas antar-klien membuat konvergensi awal melambat, namun model tetap berhasil naik dari $34.81\%$ ke **$55.62\%$** tanpa mengalami divergensi bobot (*gradient explosion*).

---

## 4. Implementasi Kode Program
* **Skrip Mandiri**: [`non_iid_analysis.py`](file:///c:/WAR-FL/Documentation/System_4_Management_and_Audit/4_3_Worker_Monitor_and_Non_IID_Analysis/non_iid_analysis.py)
* **Rujukan Notebook Utama**: [`main.ipynb`](file:///c:/WAR-FL/main.ipynb) pada **Cell 77 & Cell 78**.

---

## 5. File Output yang Dihasilkan
* **Lokasi Visualisasi Output**: [`fig_7_non_iid_analysis.png`](file:///c:/WAR-FL/Documentation/System_4_Management_and_Audit/4_3_Worker_Monitor_and_Non_IID_Analysis/fig_7_non_iid_analysis.png)
  * Kurva konvergensi akurasi vs ronde komunikasi pada 3 spektrum heterogenitas $\alpha \in \{0.1, 0.5, 1.0\}$.

---

## 6. Korelasi dengan Naskah Paper (`paper_draft_filled.md`)
* **Section V-F (Non-IID and Ablation Analysis, Line 225)**: Pembahasan detail mengenai ketahanan FL-WeAR pada kondisi statistical heterogeneity.
* **Fig. 7 (Non-IID Convergence Analysis: Accuracy vs Round)**: Visualisasi grafis kurva konvergensi non-IID.
