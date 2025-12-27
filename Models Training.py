# mfcc_multi_model.py
# Modifikasi: Training 3 Model (RF, KNN, SVM) + Export + Gambar Confusion Matrix

import pandas as pd
import numpy as np
from collections import Counter
import sys
import os
import joblib
import warnings

# Library Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Library Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

file_path = 'extracted_mfcc_optimized.csv'
artifacts_dir = "./artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# --- 1. Load data ---
try:
    data = pd.read_csv(file_path)
    print(f"Loaded '{file_path}' — rows: {len(data)}, cols: {len(data.columns)}")
    if data.empty:
        sys.exit("Error: CSV kosong.")
except FileNotFoundError:
    sys.exit(f"Error: File '{file_path}' tidak ditemukan. Pastikan file ada di folder yang sama.")
except Exception as e:
    sys.exit(f"Error membaca file: {e}")

# --- 2. split X,y and encode labels ---
X = data.iloc[:, :-1].values
y_raw = data.iloc[:, -1].values

if X.size == 0 or y_raw.size == 0:
    sys.exit("Error: fitur atau label kosong.")

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_
print(f"Label encoded: {len(class_names)} classes ({class_names})")

# --- 3. remove rare classes (<2 samples) ---
class_counts = Counter(y)
rare = [c for c, cnt in class_counts.items() if cnt < 2]
if rare:
    print("Removing rare classes:", rare)
    mask = ~np.isin(y, rare)
    X, y = X[mask], y[mask]

# --- 4. optional augmentation to balance classes ---
# (Duplikasi data minoritas + sedikit noise agar seimbang)
class_counts = Counter(y)
max_count = max(class_counts.values())
X_list = list(X)
y_list = list(y)

print("Melakukan augmentasi data...")
for cls, cnt in class_counts.items():
    if cnt < max_count:
        samples = X[y == cls]
        n_add = max_count - cnt
        for i in range(n_add):
            base = samples[i % len(samples)]
            noise = np.random.normal(0, 0.01, size=base.shape)
            X_list.append(base + noise)
            y_list.append(cls)

X = np.vstack(X_list)
y = np.array(y_list)
print("Distribusi kelas setelah augmentasi:", Counter(y))

# --- 5. train/test split ---
# Stratify memastikan proporsi kelas sama di train dan test
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

print(f"Data Train: {len(y_train)} | Data Test: {len(y_test)}")

# --- 6. Scaling (PENTING untuk KNN dan SVM) ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Simpan Scaler dan Label Encoder (hanya perlu sekali)
joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.joblib"))
joblib.dump(le, os.path.join(artifacts_dir, "label_encoder.joblib"))
print(f"\n[INFO] Scaler & LabelEncoder disimpan di {artifacts_dir}")

# --- 7. Definisi Model ---
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
}

# --- 8. Loop Training & Evaluasi ---
print("\n=== Mulai Training 3 Model ===")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_s, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"-> Akurasi {name}: {acc:.4f}")
    
    # Simpan Model
    filename = f"{name.lower()}_model.joblib"
    joblib.dump(model, os.path.join(artifacts_dir, filename))
    
    # --- 9. Buat & Simpan Confusion Matrix Image ---
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {name} (Acc: {acc:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Simpan gambar
    img_name = f"cm_{name.lower()}.png"
    plt.savefig(os.path.join(artifacts_dir, img_name))
    plt.close() # Tutup plot agar tidak menumpuk di memori
    print(f"-> Saved model & CM plot ({img_name})")

print(f"\nSelesai! Semua output ada di folder '{artifacts_dir}'.")