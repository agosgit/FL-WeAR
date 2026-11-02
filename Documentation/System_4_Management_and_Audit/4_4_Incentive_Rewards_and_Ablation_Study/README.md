# System 4.4: Reward Distribution Mechanism & Ablation Study

## 1. Deskripsi Komponen & Tujuan
Untuk menjaga keberlanjutan partisipasi pekerja las dalam ekosistem pembelajaran terfederasi sukarela di industri manufaktur serta memvalidasi kontribusi masing-masing modul arsitektur:
1. **Mekanisme Distribusi Reward Berbasis Kontribusi Komposit (*Section III-G Paper*)**: Memberikan insentif token/finansial yang adil bagi pekerja berdasarkan 3 dimensi objektif: kuantitas data, marginal utility (kualitas peningkatan akurasi server), dan efisiensi komunikasi (kecepatan latensi).
2. **Studi Ablasi Komprehensif (*Table VIII Paper*)**: Mengisolasi dan mengukur degradasi performa jika salah satu komponen kunci (LSTM, CNN, DP, federasi kolaboratif, atau penyeimbang kelas) dinonaktifkan.

---

## 2. Formulasi Matematis Reward Distribution Mechanism (Section III-G Paper)
Pada setiap ronde komunikasi $t$, setiap klien pekerja las $k \in \{1, \dots, K\}$ memperoleh skor kontribusi komposit terpadu $S_k^{(t)}$:

$$S_k^{(t)} = \alpha \hat{q}_k^{(t)} + \beta \widehat{\Delta A_k^{(t)}} + \gamma \hat{r}_k^{(t)}$$

di mana:
1. **Kuantitas Sampel Lokal Ternormalisasi ($\hat{q}_k^{(t)}$)**:
   $$\hat{q}_k^{(t)} = \frac{n_k}{\sum_{j=1}^K n_j}$$
2. **Peningkatan Marjinal Kualitas Model Server ($\widehat{\Delta A_k^{(t)}}$)**:
   $$\widehat{\Delta A_k^{(t)}} = \frac{\max\left(0, \Delta A_k^{(t)}\right)}{\sum_{j=1}^K \max\left(0, \Delta A_j^{(t)}\right)}$$
   *di mana $\Delta A_k^{(t)}$ adalah perbaikan performa validasi server setelah pembaruan model klien $k$ digabungkan.*
3. **Efisiensi Komunikasi / Inverse-Latency Ternormalisasi ($\hat{r}_k^{(t)}$)**:
   $$\hat{r}_k^{(t)} = \frac{1 / \tau_k}{\sum_{j=1}^K 1 / \tau_j}$$
   *di mana $\tau_k$ adalah waktu latensi transmisi & komputasi lokal klien $k$ (ms).*
4. **Bobot Komposit ($\alpha, \beta, \gamma$)**:
   $$\alpha + \beta + \gamma = 1.0 \quad (\text{contoh default: } \alpha = 0.4, \beta = 0.4, \gamma = 0.2)$$

### Akumulasi Token & Perlindungan Privasi:
Skor $S_k^{(t)}$ diakumulasikan sepanjang ronde federasi dan dikonversikan menjadi token insentif pada akhir shift kerja ($R_k = R_{pool} \times S_k$). Setiap pekerja hanya dapat melihat riwayat skor miliknya sendiri guna menghindari tekanan kompetitif di lantai pabrik.

---

## 3. Hasil Studi Ablasi Sistem (TABLE VIII Paper - Sinkron Hasil Training Nyata)

| Konfigurasi Eksperimen | Akurasi (%) | Macro F1 (%) | $\Delta$ F1 (pp) | Observasi Kunci & Dampak Arsitektur |
| :--- | :---: | :---: | :---: | :--- |
| **Full Model (Proposed FL-WeAR)** | **73.48** | **75.23** | **—** | **Keseimbangan optimal privasi & utilitas (Hasil Training Nyata Table V)** |
| **w/o LSTM (CNN only)** | 67.28 | 67.78 | **-7.45** | Kehilangan kemampuan memodelkan dinamika sekuensial temporal |
| **w/o CNN (LSTM only)** | 64.08 | 64.03 | **-11.20** | Kehilangan kemampuan mengekstrak filter pola gerak lokal |
| **w/o DP (FL only, B5)** | 77.35 | 78.09 | **+2.86** | Batas atas FL tanpa jaminan privasi matematis formal (Hasil Nyata B5) |
| **w/o FL (Local only, B4)** | 66.59 | 65.30 | **-9.93** | Bukti nyata manfaat kolaborasi lintas pekerja non-IID (Hasil Nyata B4) |
| **w/o Class Balancing** | 43.48 | 18.00 | **-57.23** | Keruntuhan performa drastis pada kelas-kelas minoritas |

---

## 4. Implementasi Kode Program
* **Skrip Mandiri**: [`incentive_and_ablation.py`](file:///c:/WAR-FL/Documentation/System_4_Management_and_Audit/4_4_Incentive_Rewards_and_Ablation_Study/incentive_and_ablation.py)
* **Rujukan Naskah Paper**: [`0.Inass_Rev5_FL_WeAR.pdf`](file:///c:/WAR-FL/0.Inass_Rev5_FL_WeAR.pdf) pada **Section III-G (Reward Distribution Mechanism)** & **TABLE VIII (Ablation Study)**.

---

## 5. File Output yang Dihasilkan
1. [`table_viii_ablation_results.csv`](file:///c:/WAR-FL/Documentation/System_4_Management_and_Audit/4_4_Incentive_Rewards_and_Ablation_Study/table_viii_ablation_results.csv): Rekapitulasi kuantitatif studi ablasi.
2. [`reward_distribution_simulation.csv`](file:///c:/WAR-FL/Documentation/System_4_Management_and_Audit/4_4_Incentive_Rewards_and_Ablation_Study/reward_distribution_simulation.csv): Hasil simulasi alokasi token insentif proporsional per pekerja.
