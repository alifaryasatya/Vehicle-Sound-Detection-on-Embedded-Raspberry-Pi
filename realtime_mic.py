import numpy as np
import librosa
import sounddevice as sd
import joblib
import sys

# --- KONFIGURASI ---
MODEL_PATH = "./artifacts/rf_model.joblib" 
SCALER_PATH = "./artifacts/scaler.joblib"
ENCODER_PATH = "./artifacts/label_encoder.joblib"

SAMPLE_RATE = 22050
DURATION = 2.0        
N_MFCC = 13           # KEMBALIKAN KE 13 (karena kita akan hitung mean+std jadi total 26)

# --- LOAD ARTIFACTS ---
print("Loading model...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    print("Model loaded successfully.")
except Exception as e:
    sys.exit(f"Error load model: {e}")

def extract_features(audio, sr):
    try:
        # 1. Ekstrak 13 MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # 2. Hitung Rata-rata (Mean) -> 13 fitur
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # 3. Hitung Standar Deviasi (Std) -> 13 fitur
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # 4. Gabungkan jadi 26 fitur (Sesuai urutan di CSV: Mean dulu, baru Std)
        combined_features = np.hstack((mfccs_mean, mfccs_std))
        
        return combined_features
    except Exception as e:
        print(f" (Ext. Err: {e})", end="")
        return None

def live_prediction():
    device_info = sd.query_devices(kind='input')
    print(f"\n[INFO] Mic: {device_info['name']}")
    
    print("\n" + "="*50)
    print(f"Mulai! Menggunakan {N_MFCC} MFCC (Mean+Std = {N_MFCC*2} Fitur).")
    print("Silakan bersuara dekat mic...")
    print("="*50 + "\n")

    try:
        while True:
            # Rekam
            recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                               samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            
            audio_data = recording.flatten()
            
            # Cek Volume
            volume = np.max(np.abs(audio_data))
            
            if volume < 0.005:
                print(f"Volume: {volume:.4f} (Hening)     ", end="\r")
                continue
            
            print(f"Volume: {volume:.4f} | ", end="")

            # Ekstrak
            features = extract_features(audio_data, SAMPLE_RATE)
            
            if features is not None:
                # Cek dimensi fitur (Harus 26)
                expected_features = N_MFCC * 2
                if features.shape[0] != expected_features:
                    print(f"Error: Fitur {features.shape[0]}, harusnya {expected_features}")
                    continue

                # Scaling
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Prediksi
                pred = model.predict(features_scaled)[0]
                label = le.inverse_transform([pred])[0]
                
                print(f"Prediksi: >>> {str(label)} <<<") 
                
            else:
                print("Gagal ekstrak fitur.")

    except KeyboardInterrupt:
        print("\nProgram dihentikan.")
    except Exception as e:
        print(f"\nError Runtime: {e}")

if __name__ == "__main__":
    live_prediction()