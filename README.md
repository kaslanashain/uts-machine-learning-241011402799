# Tugas UTS Machine Learning: Linear Regression & Logistic Regression

Repository ini berisi tugas Ujian Tengah Semester mata kuliah **Machine Learning**. Tugas mencakup implementasi **Linear Regression** (dengan Normal Equation dan Gradient Descent) serta **Logistic Regression** (dengan Gradient Descent) dari nol menggunakan NumPy, lalu membandingkannya dengan scikit-learn sebagai bonus.

## Identitas Mahasiswa

| Keterangan | Isi |
|---|---|
| **Nama** | Muhammad Fadhli Al Akhbar |
| **NIM** | 241011402799 |
| **Kelas** | 05TPLM003 |
| **Mata Kuliah** | Machine Learning |
| **`RANDOM_STATE`** | 99 (2 digit terakhir NIM: 2410114027**99**) |

## Struktur Repository



## Tujuan Pembelajaran

- Memahami konsep dasar regresi linear dan regresi logistik.
- Mengimplementasikan Normal Equation dan Gradient Descent secara manual.
- Memahami pengaruh learning rate terhadap konvergensi.
- Mengevaluasi model regresi dan klasifikasi menggunakan metrik yang sesuai.
- Memahami konsep threshold, confusion matrix, dan decision boundary.

## Ringkasan Materi

### Bagian A: Linear Regression (35%)

**Kasus:** Sebuah toko online ingin mengetahui pengaruh **biaya iklan** terhadap **penjualan** setiap bulan. Dataset `Ads_Data.csv` berisi 40 baris.

| Soal | Materi |
|---|---|
| A1 | Memuat data, memisahkan `X` (BiayaIklan, berbentuk `(n, 1)`) dan `y` (Penjualan), serta membuat scatter plot. |
| A2 | Membagi data dengan fungsi `train_test_split` buatan sendiri (`test_size=0.2`), menghasilkan 32 data training dan 8 data testing. |
| A3 | Training dengan **Normal Equation**: θ = (XᵀX)⁻¹Xᵀy. Hasil: intercept ≈ 29.1333 dan koefisien ≈ 8.9200. |
| A4 | Evaluasi dengan MAE, MSE, RMSE, dan R². R² training ≈ 0.9503 dan R² testing ≈ 0.8658. |
| A5 | **Gradient Descent** dengan tiga learning rate (0.001; 0.02; 0.1) selama 5000 iterasi. |
| A6 | Visualisasi garis regresi dan prediksi untuk biaya iklan baru (4.5; 7; dan 12 juta). |

**Temuan penting:**
- Hubungan biaya iklan dan penjualan bersifat **positif dan cukup linear**.
- Model memiliki R² testing yang sedikit lebih rendah daripada training, wajar untuk data baru.
- **Learning rate 0.001** terlalu kecil sehingga belum konvergen dalam 5000 iterasi.
- **Learning rate 0.02** menghasilkan parameter yang mendekati baseline Normal Equation.
- **Learning rate 0.1** terlalu besar dan menyebabkan **divergensi** (overflow, nilai `NaN`/`inf`).
- Prediksi untuk biaya iklan 12 juta merupakan **ekstrapolasi** karena berada di luar rentang data, sehingga tingkat ketidakpastiannya tinggi.

**Prediksi A6:**

| Biaya Iklan | Prediksi Penjualan | Keterangan |
|---|---|---|
| 4.5 juta | 69.27 juta | Di dalam rentang data — cukup dipercaya |
| 7.0 juta | 91.57 juta | Di dalam rentang data — cukup dipercaya |
| 12.0 juta | 136.17 juta | **Ekstrapolasi** — ketidakpastian tinggi |

### Bagian B: Logistic Regression (40%)

**Kasus:** Filter spam berdasarkan **jumlah link** dan **jumlah kata promosi** pada `Email_Data.csv` (60 email; 35 bukan spam dan 25 spam).

| Soal | Materi |
|---|---|
| B1 | Memuat data dan membuat scatter plot dua fitur dengan warna per kelas. |
| B2 | Split data (`test_size=0.25` → 45 train / 15 test), lalu training Logistic Regression dengan Gradient Descent (`lr=0.1`, 10000 iterasi) menggunakan fungsi sigmoid dan log loss. |
| B3 | Evaluasi dengan confusion matrix, accuracy, precision, recall, dan F1-score. |
| B4 | Visualisasi **decision boundary** dan kontur probabilitas spam. |
| B5 | Eksperimen **threshold** (0.3; 0.5; 0.7). Threshold final yang dipilih: **0.7** (dipakai di B6). |
| B6 | Prediksi kelas untuk empat email baru menggunakan threshold 0.7. |

**Hasil model (threshold 0.5 pada data testing):**

| Metrik | Nilai |
|---|---|
| Confusion Matrix `[[TN, FP], [FN, TP]]` | `[[11, 0], [1, 3]]` |
| Accuracy | 0.9333 |
| Precision | 1.0000 |
| Recall | 0.7500 |
| F1-Score | 0.8571 |

**Temuan penting:**
- Koefisien kedua fitur **positif** (Link = 0.7549; Promo = 0.8160), artinya semakin banyak link dan kata promo, semakin tinggi probabilitas email tersebut spam.
- **`KataPromo` lebih dominan** daripada `JumlahLink` (koefisien 0.8160 > 0.7549).
- Log loss menurun stabil selama training, menandakan konvergensi.
- Pada data testing, **False Positive lebih merugikan** daripada False Negative karena email penting bisa salah masuk folder spam.
- Pada eksperimen ini, threshold 0.3, 0.5, dan 0.7 menghasilkan confusion matrix yang sama (TN=11, FP=0, FN=1, TP=3). Perubahan threshold belum terlihat dampaknya karena data testing yang kecil dan probabilitas kelas sudah terpisah jelas.

**Prediksi B6 (threshold = 0.7):**

| Email | Link | Promo | Prob(Spam) | Prediksi |
|---|---|---|---|---|
| E1 | 1 | 2 | 0.0009 | Bukan Spam |
| E2 | 5 | 6 | 0.3334 | **Bukan Spam** (di bawah threshold 0.7) |
| E3 | 9 | 10 | 0.9963 | Spam |
| E4 | 2 | 11 | 0.7544 | **Spam** (didominasi fitur Promo) |

**Insight B6:** Meskipun E2 punya cukup banyak link dan kata promo, ia tetap diprediksi **Bukan Spam** karena threshold dipilih konservatif (0.7). Sebaliknya, **E4 hanya punya 2 link tetapi 11 kata promo** dan tetap diklasifikasikan Spam — menunjukkan bahwa fitur `KataPromo` lebih berpengaruh dalam model ini.

### Bagian C: Pemahaman Konsep (15%)

Membahas konsep teoretis:

1. **C1** — Mengapa Logistic Regression termasuk algoritma **klasifikasi** meskipun namanya mengandung "regression" (fungsi sigmoid memetakan nilai linear ke probabilitas 0–1).
2. **C2** — Perbedaan **Normal Equation** (solusi langsung, cepat untuk data kecil, kompleksitas O(n³)) dan **Gradient Descent** (iteratif, cocok untuk data besar dan model kompleks).
3. **C3** — Pentingnya evaluasi pada **data testing** untuk mendeteksi **overfitting** dan mengukur generalisasi.
4. **C4** — Perbedaan fungsi loss: **MSE** untuk output kontinu dan **log loss** untuk output probabilitas kelas.
5. **C5** — Alasan hasil bisa berbeda antarmahasiswa (perbedaan `random_state`, learning rate, iterasi, dan inisialisasi bobot).

### Bonus (+5): Perbandingan dengan scikit-learn

- **Linear Regression:** koefisien dan R² dari implementasi custom dibandingkan dengan `sklearn.linear_model.LinearRegression`. Hasilnya **identik** karena keduanya menyelesaikan masalah OLS yang sama (b = 29.1333, m = 8.9200; R² train = 0.9503, test = 0.8658).
- **Logistic Regression:** parameter dan akurasi dibandingkan dengan `LogisticRegression` scikit-learn (solver `lbfgs`, `penalty=None`). Parameter bisa sedikit berbeda karena perbedaan algoritma optimasi, **tetapi prediksi kelas pada data testing identik** (`True`) dengan akurasi yang sama (0.9333).

## Catatan Teknis

- **Split data berbeda antar bagian:**
  - Bagian A: `test_size=0.2` → 32 train / 8 test
  - Bagian B: `test_size=0.25` → 45 train / 15 test
- **Threshold final Bagian B:** 0.7 (dipakai di B6).
- **Warning overflow di A5 (lr=0.1) disengaja** untuk demonstrasi divergensi Gradient Descent, bukan bug.
- Seluruh fungsi inti (train-test split, Normal Equation, Gradient Descent, metrik evaluasi) diimplementasikan manual dengan NumPy. scikit-learn hanya digunakan pada bagian bonus.
- Seed `np.random.seed(99)` dan `random_state=99` dipakai agar hasil dapat di reproduksi
