import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
import sys

# === 1. MEMUAT DATA ===
file_path = 'extracted_mfcc_optimized.csv'
try:
    data = pd.read_csv(file_path)
    print(f"File '{file_path}' berhasil dimuat. Total {len(data)} baris, {len(data.columns)} kolom.")
    if data.empty:
        sys.exit("❌ Error: File data kosong.")
except FileNotFoundError:
    sys.exit(f"❌ Error: File '{file_path}' tidak ditemukan.")
except Exception as e:
    sys.exit(f"❌ Error saat membaca file: {e}")

# === 2. PEMISAHAN FITUR & LABEL ===
try:
    X = data.iloc[:, :-1]
    y_raw = data.iloc[:, -1]
    if X.empty or y_raw.empty:
        raise ValueError("Fitur (X) atau label (y) kosong setelah pemisahan.")
    print(f"Pemisahan fitur (X) dan label (y) berhasil. X: {X.shape}, y: {y_raw.shape}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Label berhasil di-encode: {len(np.unique(y))} kelas unik ditemukan.")
except Exception as e:
    sys.exit(f"❌ Error saat memisahkan atau encoding data: {e}")

# === 3. DETEKSI DAN HAPUS KELAS LANGKA ===
class_counts = Counter(y)
rare_classes = [cls for cls, count in class_counts.items() if count < 2]
if rare_classes:
    print(f"⚠ Menghapus kelas langka (jumlah < 2): {rare_classes}")
    mask = ~np.isin(y, rare_classes)
    X, y = X[mask], y[mask]
    print(f"Jumlah data baru setelah penghapusan: {len(y)}")

# === 4. PEMBAGIAN DATA ===
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    print("⚠ Stratify dinonaktifkan karena distribusi kelas tidak seimbang.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

print(f"Data berhasil dibagi: {len(y_train)} data latih, {len(y_test)} data validasi.")

# === 5. PENSKALAAN FITUR ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Penskalaan fitur (StandardScaler) berhasil diterapkan.")

# === 6. FUNGSI EVALUASI MODEL ===
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print("\n" + "="*45)
    print(f"Model: {name}")
    print("="*45)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Akurasi {name}: {acc:.3f}")

    # === Plot confusion matrix ===
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# === 7. PELATIHAN DAN EVALUASI MODEL ===
print("\n===============================")
print("🔹 MODEL DENGAN PENSKALAAN FITUR")
print("===============================")

evaluate_model("K-Nearest Neighbors (KNN)", KNeighborsClassifier(n_neighbors=5), 
                X_train_scaled, X_test_scaled, y_train, y_test)

evaluate_model("Support Vector Machine (SVM)", SVC(kernel='rbf', C=1.0, random_state=42), 
                X_train_scaled, X_test_scaled, y_train, y_test)

evaluate_model("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42), 
                X_train_scaled, X_test_scaled, y_train, y_test)

print("\n✅ Semua model telah dilatih dan divisualisasikan (versi dengan skala).")