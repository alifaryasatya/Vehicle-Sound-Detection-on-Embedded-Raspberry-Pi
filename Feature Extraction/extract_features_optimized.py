import librosa
import numpy as np
import pandas as pd
import os
import time

# --- Konfigurasi Ekstraksi ---
N_MFCC = 40        # Jumlah koefisien MFCC yang akan diekstrak
# MAX_PAD_LEN dihapus karena tidak lagi diperlukan
# ------------------------------

def extract_mfcc_features_optimized(file_path, n_mfcc, base_dir="./"):
    """
    Mengekstrak fitur MFCC menggunakan agregasi statistik (mean dan std dev).
    """
    full_path = os.path.join(base_dir, file_path)
    
    try:
        # 1. Muat sinyal suara
        audio, sr = librosa.load(full_path, sr=None) 
        
        # 2. Ekstraksi Fitur MFCC
        # librosa akan menghasilkan matriks (N_MFCC, N_Frames)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # 3. Agregasi Statistik (PENGGANTI NORMALISASI PANJANG & FLATTENING)
        # Hitung Mean (rata-rata) untuk setiap koefisien MFCC sepanjang waktu (axis=1)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Hitung Standard Deviation (deviasi standar) untuk setiap koefisien MFCC
        mfcc_std = np.std(mfccs, axis=1)
        
        # Gabungkan mean dan std dev menjadi satu vektor panjang (40 + 40 = 80 fitur)
        features_combined = np.hstack([mfcc_mean, mfcc_std])
        
    except Exception as e:
        print(f"Error memproses {full_path}: {e}")
        return None
    
    return features_combined

# --- Fungsi Utama ---
def process_dataset(csv_file, output_csv, base_audio_dir):
    start_time = time.time()
    print(f"Memulai proses ekstraksi fitur dari {csv_file} (Metode Agregasi Statistik)...")
    
    # Memuat file metadata
    df_metadata = pd.read_csv(csv_file)
    
    # List untuk menyimpan semua fitur yang diekstrak
    all_features = []
    
    # Iterasi melalui setiap baris dalam metadata
    for index, row in df_metadata.iterrows():
        file_path = row['file_path']
        label = row['class']
        
        # Ekstraksi fitur
        features = extract_mfcc_features_optimized(
            file_path, 
            N_MFCC, 
            base_dir=base_audio_dir
        )
        
        if features is not None:
            # Mengubah array fitur menjadi list dan menambahkan label
            feature_row = features.tolist()
            feature_row.append(label)
            all_features.append(feature_row)
        
        # Menampilkan progres setiap 100 file
        if (index + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Proses: {index + 1}/{len(df_metadata)} file selesai. Waktu: {elapsed:.2f} detik.")

    # Membuat DataFrame hasil
    
    # Jumlah fitur baru: N_MFCC (mean) + N_MFCC (std dev)
    num_features = N_MFCC * 2 
    
    # Membuat nama kolom yang lebih jelas (mfcc_1_mean, mfcc_1_std, mfcc_2_mean, ...)
    feature_cols = []
    for i in range(1, N_MFCC + 1):
        feature_cols.append(f'mfcc_{i}_mean')
    for i in range(1, N_MFCC + 1):
        feature_cols.append(f'mfcc_{i}_std')
        
    df_features = pd.DataFrame(all_features, columns=feature_cols + ['label'])
    
    # Menyimpan ke file CSV baru
    df_features.to_csv(output_csv, index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n-------------------------------------------")
    print(f"✅ Ekstraksi selesai. Total waktu: {total_time:.2f} detik.")
    print(f"✅ Data fitur disimpan ke {output_csv}")
    print(f"✅ Jumlah sampel yang diproses: {len(df_features)}")
    print(f"✅ Jumlah total fitur per sampel: {num_features} (Jauh lebih efisien!)")
    print("-------------------------------------------")

# --- Eksekusi Script ---
if __name__ == "__main__":
    # Ubah sesuai dengan lokasi file Anda:
    METADATA_CSV = 'labels_with_additional_data.csv'
    OUTPUT_CSV_FILE = 'extracted_mfcc_optimized.csv' # Nama file baru
    
    # PENTING: Ganti BASE_AUDIO_DIR dengan path folder Anda
    BASE_AUDIO_DIR = r'C:\Tugas\soumdpro\vehicle_type_sound_dataset' 
    
    process_dataset(METADATA_CSV, OUTPUT_CSV_FILE, BASE_AUDIO_DIR)