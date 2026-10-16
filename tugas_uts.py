# Export kode dari tugas_uts_revisi.ipynb
# Catatan: sel magic Jupyter seperti %pip perlu dijalankan di Jupyter, bukan Python biasa.

# ============================================================
# AUTO-INSTALL LIBRARY (jalankan sekali di awal)
# ============================================================
import sys
print("Kernel memakai Python di:", sys.executable)
# %pip install --quiet pandas numpy matplotlib scikit-learn
print("Selesai install. Restart kernel lalu Run All.")

# ============================================================
# UTS MACHINE LEARNING - LINEAR & LOGISTIC REGRESSION
# Nama   : Muhammad Fadhli Al Akhbar
# NIM    : 241011402799
# Kelas  : 05TPLM003
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# scikit-learn HANYA dipakai untuk soal bonus
from sklearn.linear_model import LinearRegression as SkLearnLR
from sklearn.linear_model import LogisticRegression as SkLearnLogReg
from sklearn.metrics import accuracy_score as sk_acc, r2_score as sk_r2

# random_state = 2 digit terakhir NIM (241011402799 -> 99)
RANDOM_STATE = 99
np.random.seed(RANDOM_STATE)

print("=" * 60)
print(" INISIALISASI DATA & FUNGSI UTAMA")
print("=" * 60)

# ============================================================
# FUNGSI TRAIN_TEST_SPLIT MANUAL
# ============================================================
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ---------- A1: Load & Visualisasi ----------
ads_df = pd.read_csv('Ads_Data.csv')
print("\n[A1] 5 Data Pertama Ads_Data:")
print(ads_df.head())
print(f"[A1] Jumlah baris data: {ads_df.shape[0]}")

X_ads = ads_df[['BiayaIklan']].values  # (n, 1)
y_ads = ads_df['Penjualan'].values

plt.figure(figsize=(8, 5))
plt.scatter(X_ads, y_ads, color='blue', alpha=0.7)
plt.title('Scatter Plot: Biaya Iklan vs Penjualan')
plt.xlabel('Biaya Iklan (juta Rp)')
plt.ylabel('Penjualan (juta Rp)')
plt.grid(True)
plt.show()

# ---------- A2: Split Data ----------
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_ads, y_ads, test_size=0.2, random_state=RANDOM_STATE
)
print("\n[A2] Shape data:")
print(f"  X_train: {X_train_a.shape}")
print(f"  X_test : {X_test_a.shape}")
print(f"  y_train: {y_train_a.shape}")
print(f"  y_test : {y_test_a.shape}")

# ---------- A3: Normal Equation ----------
class LinearRegressionNormal:
    """Linear Regression dengan Normal Equation: theta = (X^T X)^-1 X^T y"""
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.r_[self.intercept_, self.coef_]
        return X_b @ theta


lr_normal = LinearRegressionNormal().fit(X_train_a, y_train_a)
print("\n[A3] Hasil Normal Equation:")
print(f"  Intercept (b): {lr_normal.intercept_:.4f}")
print(f"  Koefisien (m): {lr_normal.coef_[0]:.4f}")
print(f"  Persamaan    : y_hat = {lr_normal.intercept_:.4f} + {lr_normal.coef_[0]:.4f} * X")

# ---------- A4: Evaluasi Model ----------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


y_pred_train = lr_normal.predict(X_train_a)
y_pred_test  = lr_normal.predict(X_test_a)

df_eval = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R2 Score'],
    'Training': [mae(y_train_a, y_pred_train), mse(y_train_a, y_pred_train),
                 rmse(y_train_a, y_pred_train), r2_score(y_train_a, y_pred_train)],
    'Testing':  [mae(y_test_a, y_pred_test), mse(y_test_a, y_pred_test),
                 rmse(y_test_a, y_pred_test), r2_score(y_test_a, y_pred_test)]
})
print("\n[A4] Tabel Evaluasi Model Regresi:")
print(df_eval.to_string(index=False))

# ============================================================
# Aktifkan warning runtime HANYA untuk eksperimen A5
# agar RuntimeWarning (overflow) muncul di output
# ============================================================
import warnings
warnings.filterwarnings('default', category=RuntimeWarning)
print("Warning runtime AKTIF untuk eksperimen A5")

# ---------- A5: Gradient Descent & Learning Rate ----------
class LinearRegressionGD:
    """Linear Regression dengan Gradient Descent"""
    def __init__(self, n_iterations=5000, lr=0.01):
        self.n_iterations = n_iterations
        self.lr = lr
        self.loss_history_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.theta = np.zeros(X.shape[1] + 1)
        X_b = np.c_[np.ones((n_samples, 1)), X]
        for i in range(self.n_iterations):
            gradients = (2 / n_samples) * X_b.T @ (X_b @ self.theta - y)
            self.theta -= self.lr * gradients
            loss = np.mean((X_b @ self.theta - y) ** 2)
            self.loss_history_.append(loss)
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self


print("\n[A5] Hasil Eksperimen Gradient Descent (n_iterations=5000):")
print(f"  Baseline Normal Equation -> Intercept: {lr_normal.intercept_:.4f}, Coef: {lr_normal.coef_[0]:.4f}")
learning_rates = [0.001, 0.02, 0.1]
hasil_gd = []
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    model_gd = LinearRegressionGD(n_iterations=5000, lr=lr).fit(X_train_a, y_train_a)
    loss_akhir = model_gd.loss_history_[-1]
    print(f"  LR = {lr:<5} -> Intercept: {model_gd.intercept_:8.4f}, "
          f"Coef: {model_gd.coef_[0]:8.4f}, Loss Akhir: {loss_akhir:12.4f}")
    hasil_gd.append({
        "Metode": f"Gradient Descent (lr={lr})",
        "Learning Rate": lr,
        "Intercept": model_gd.intercept_,
        "Koefisien": model_gd.coef_[0],
        "Loss Akhir": loss_akhir,
        "Status": ("Divergen / tidak finite" if not np.isfinite(loss_akhir)
                   else "Mendekati baseline" if lr == 0.02
                   else "Belum konvergen / perlu dibandingkan")
    })

    loss_arr = np.array(model_gd.loss_history_, dtype=float)
    loss_arr = np.where(np.isfinite(loss_arr), loss_arr, np.nan)
    plt.plot(loss_arr, label=f'lr = {lr}')

plt.yscale('log')
plt.title('Loss (MSE) vs Iterations untuk Berbagai Learning Rate')
plt.xlabel('Iteration')
plt.ylabel('MSE Loss (Log Scale)')
plt.legend()
plt.grid(True)
plt.show()

print("\n[A5] Tabel perbandingan numerik:")
df_gd = pd.DataFrame(hasil_gd)
print(df_gd.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# Matikan kembali warning runtime untuk sel-sel berikutnya
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
print("Warning runtime dimatikan kembali")

# ---------- A6: Visualisasi & Prediksi ----------
plt.figure(figsize=(8, 5))
plt.scatter(X_test_a, y_test_a, color='red', label='Data Testing', alpha=0.8)
X_line = np.linspace(X_ads.min(), X_ads.max(), 100).reshape(-1, 1)
plt.plot(X_line, lr_normal.predict(X_line), color='blue', linewidth=2, label='Garis Regresi')
plt.title('Data Testing & Garis Regresi (Normal Equation)')
plt.xlabel('Biaya Iklan (juta Rp)')
plt.ylabel('Penjualan (juta Rp)')
plt.legend()
plt.grid(True)
plt.show()

new_ads = np.array([[4.5], [7.0], [12.0]])
preds_ads = lr_normal.predict(new_ads)
print("\n[A6] Prediksi Penjualan untuk Biaya Iklan Baru:")
for cost, pred in zip(new_ads.flatten(), preds_ads):
    print(f"  Biaya Iklan {cost:>4} juta -> Prediksi Penjualan: {pred:.2f} juta Rp")

# ---------- B1: Load & Visualisasi ----------
email_df = pd.read_csv('Email_Data.csv')
print("\n[B1] 5 Data Pertama Email_Data:")
print(email_df.head())
print("\n[B1] Distribusi Spam:")
print(email_df['Spam'].value_counts().to_dict())

X_email = email_df[['JumlahLink', 'KataPromo']].values
y_email = email_df['Spam'].values

plt.figure(figsize=(8, 5))
plt.scatter(X_email[y_email == 0, 0], X_email[y_email == 0, 1],
            color='blue', label='Bukan Spam (0)', alpha=0.8)
plt.scatter(X_email[y_email == 1, 0], X_email[y_email == 1, 1],
            color='red', label='Spam (1)', alpha=0.8)
plt.title('Scatter Plot: Jumlah Link vs Kata Promo')
plt.xlabel('Jumlah Link')
plt.ylabel('Kata Promo')
plt.legend()
plt.grid(True)
plt.show()

# ---------- B2: Split & Training ----------
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_email, y_email, test_size=0.25, random_state=RANDOM_STATE
)


class LogisticRegressionGD:
    """Logistic Regression dengan Gradient Descent"""
    def __init__(self, n_iterations=10000, lr=0.1):
        self.n_iterations = n_iterations
        self.lr = lr
        self.loss_history_ = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features + 1)
        X_b = np.c_[np.ones((n_samples, 1)), X]

        for i in range(self.n_iterations):
            y_pred = self.sigmoid(X_b @ self.theta)
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred_clipped) +
                            (1 - y) * np.log(1 - y_pred_clipped))
            self.loss_history_.append(loss)

            gradients = (1 / n_samples) * X_b.T @ (y_pred - y)
            self.theta -= self.lr * gradients

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_b @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


log_reg = LogisticRegressionGD(n_iterations=10000, lr=0.1).fit(X_train_b, y_train_b)
b  = log_reg.intercept_
m1 = log_reg.coef_[0]
m2 = log_reg.coef_[1]

print("\n[B2] Hasil Training Logistic Regression:")
print(f"  Intercept (b)             : {b:.4f}")
print(f"  Koefisien m1 (JumlahLink) : {m1:.4f}")
print(f"  Koefisien m2 (KataPromo)  : {m2:.4f}")
print(f"  Persamaan: P(Spam) = sigmoid({b:.4f} + {m1:.4f}*Link + {m2:.4f}*Promo)")

plt.figure(figsize=(8, 5))
plt.plot(log_reg.loss_history_, color='purple')
plt.title('Log Loss vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.grid(True)
plt.show()

# ---------- B3: Evaluasi Model ----------
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1, 1], cm[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1, 1], cm[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


y_pred_b = log_reg.predict(X_test_b, threshold=0.5)
cm = confusion_matrix(y_test_b, y_pred_b)

print("\n[B3] Confusion Matrix [ [TN, FP], [FN, TP] ]:")
print(cm)
print(f"  Accuracy : {accuracy_score(y_test_b, y_pred_b):.4f}")
print(f"  Precision: {precision_score(y_test_b, y_pred_b):.4f}")
print(f"  Recall   : {recall_score(y_test_b, y_pred_b):.4f}")
print(f"  F1-Score : {f1_score(y_test_b, y_pred_b):.4f}")

# ---------- B4: Decision Boundary ----------
print(f"\n[B4] Persamaan Decision Boundary:")
print(f"  x2 = -({b:.4f} + {m1:.4f} * x1) / {m2:.4f}")

x_min, x_max = X_email[:, 0].min() - 1, X_email[:, 0].max() + 1
y_min, y_max = X_email[:, 1].min() - 1, X_email[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = log_reg.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.6)
plt.colorbar(label='Probabilitas Spam')

plt.scatter(X_test_b[y_test_b == 0, 0], X_test_b[y_test_b == 0, 1],
            color='blue', label='Bukan Spam (true)', edgecolor='k', s=60)
plt.scatter(X_test_b[y_test_b == 1, 0], X_test_b[y_test_b == 1, 1],
            color='red', label='Spam (true)', edgecolor='k', s=60)

x1_vals = np.array([x_min, x_max])
x2_vals = -(b + m1 * x1_vals) / m2
plt.plot(x1_vals, x2_vals, 'k--', linewidth=2, label='Decision Boundary')

salah = y_pred_b != y_test_b
plt.scatter(X_test_b[salah, 0], X_test_b[salah, 1],
            facecolors='none', edgecolors='lime', s=250, linewidths=2.5,
            label=f'Salah Klasifikasi ({salah.sum()} titik)')

plt.title('Decision Boundary & Probabilitas Contour')
plt.xlabel('Jumlah Link')
plt.ylabel('Kata Promo')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

print("\n[B4] Data testing yang salah diklasifikasikan:")
if salah.sum() == 0:
    print("  Tidak ada data yang salah diklasifikasikan.")
else:
    for i in np.where(salah)[0]:
        print(f"  Index {i:>2}: Link={X_test_b[i,0]}, Promo={X_test_b[i,1]}, "
              f"True={y_test_b[i]}, Pred={y_pred_b[i]}")

# ---------- B5: Eksperimen Threshold ----------
thresholds = [0.3, 0.5, 0.7]
results_th = []

for th in thresholds:
    preds = log_reg.predict(X_test_b, threshold=th)
    cmat = confusion_matrix(y_test_b, preds)
    tn, fp, fn, tp = cmat[0, 0], cmat[0, 1], cmat[1, 0], cmat[1, 1]
    acc = (tp + tn) / len(y_test_b)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_v = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    results_th.append({
        'Threshold': th, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1_v
    })

df_th = pd.DataFrame(results_th)
print("\n[B5] Tabel Eksperimen Threshold:")
print(df_th.to_string(index=False))

THRESHOLD_PILIHAN = 0.7
print(f"\n[B5] Threshold yang dipilih: {THRESHOLD_PILIHAN}")
# Rentang probabilitas membantu menjelaskan hasil eksperimen threshold.
prob_test_b = log_reg.predict_proba(X_test_b)
print("\n[B5] Rentang probabilitas pada data testing:")
for kelas, nama in [(0, "Bukan Spam"), (1, "Spam")]:
    p_kelas = prob_test_b[y_test_b == kelas]
    if len(p_kelas) > 0:
        print(f"  {nama}: min={p_kelas.min():.4f}, max={p_kelas.max():.4f}")

# ---------- B6: Prediksi Email Baru ----------
new_emails = np.array([[1, 2], [5, 6], [9, 10], [2, 11]])
probs = log_reg.predict_proba(new_emails)
preds_email = log_reg.predict(new_emails, threshold=THRESHOLD_PILIHAN)

print(f"\n[B6] Prediksi Email Baru (threshold = {THRESHOLD_PILIHAN}):")
for i, (feat, prob, pred) in enumerate(zip(new_emails, probs, preds_email), 1):
    label_text = 'Spam' if pred == 1 else 'Bukan Spam'
    print(f"  Email E{i} (Link={feat[0]}, Promo={feat[1]:>2}): "
          f"Prob(Spam)={prob:.4f} -> {label_text}")

# ============================================================
# BONUS - PERBANDINGAN DENGAN SCIKIT-LEARN
# ============================================================
print("\n" + "=" * 60)
print(" BONUS - PERBANDINGAN DENGAN SCIKIT-LEARN")
print("=" * 60)

# ---- Linear Regression ----
sk_lr = SkLearnLR().fit(X_train_a, y_train_a)
print("\nLinear Regression:")
print(f"  Custom       : b = {lr_normal.intercept_:.4f}, m = {lr_normal.coef_[0]:.4f}")
print(f"  scikit-learn : b = {sk_lr.intercept_:.4f}, m = {sk_lr.coef_[0]:.4f}")
print(f"  R2 custom    : train={r2_score(y_train_a, lr_normal.predict(X_train_a)):.4f}, "
      f"test={r2_score(y_test_a, lr_normal.predict(X_test_a)):.4f}")
print(f"  R2 sklearn   : train={sk_r2(y_train_a, sk_lr.predict(X_train_a)):.4f}, "
      f"test={sk_r2(y_test_a, sk_lr.predict(X_test_a)):.4f}")

# ---- Logistic Regression ----
sk_log = SkLearnLogReg(penalty=None, solver='lbfgs', max_iter=10000).fit(X_train_b, y_train_b)
y_pred_sk = sk_log.predict(X_test_b)

print("\nLogistic Regression:")
print(f"  Custom       : b = {log_reg.intercept_:.4f}, coef = {log_reg.coef_}")
print(f"  scikit-learn : b = {sk_log.intercept_[0]:.4f}, coef = {sk_log.coef_[0]}")
print(f"  Akurasi custom      : {accuracy_score(y_test_b, y_pred_b):.4f}")
print(f"  Akurasi scikit-learn: {sk_acc(y_test_b, y_pred_sk):.4f}")
print(f"  Prediksi kelas identik: {np.array_equal(y_pred_b, y_pred_sk)}")

print("\n" + "=" * 60)
print(" PROSES PERBANDINGAN MODEL SELESAI")
print("=" * 60)

