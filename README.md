# Vehicle Sound Detection on Embedded Raspberry Pi

![Python](https://img.shields.io/badge/Language-Python%203-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red)
![ML](https://img.shields.io/badge/AI-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)


## 📌 Overview
Sistem deteksi dan klasifikasi suara kendaraan berbasis **Embedded Machine Learning** yang dirancang untuk beroperasi pada Raspberry Pi. Sistem ini memanfaatkan ekstraksi fitur audio **MFCC (Mel-Frequency Cepstral Coefficients)** untuk membedakan jenis kendaraan *(Mobil, Motor, Truk)* secara *real-time* berdasarkan karakteristik akustik mesinnya.

Berbeda dengan sistem berbasis visi komputer, pendekatan berbasis audio ini lebih efisien secara komputasi dan tetap efektif dalam kondisi pencahayaan rendah atau malam hari.

## 🎯 Tujuan Proyek
- **Real-time Monitoring**: Mengklasifikasikan jenis kendaraan yang lewat secara langsung di perangkat *edge*.
- **High Accuracy**: Mencapai akurasi klasifikasi tinggi menggunakan algoritma *Machine Learning* yang teruji.
- **Embedded Implementation**: Mengoptimalkan pemrosesan sinyal suara agar berjalan lancar pada perangkat dengan sumber daya terbatas (Raspberry Pi).

## 📊 Diagram Alur Sistem
~~~mermaid
graph LR
    A[Mic USB] -->|Raw Audio| B[Pre-processing]
    B -->|Ekstraksi MFCC| C{ML Model}
    C -->|Klasifikasi| D[Output LCD/Log]

    subgraph "Machine Learning Core"
    C
    end
~~~

## 🛠️ Spesifikasi Teknis
- **Microcontroller/SBC**: Raspberry Pi 4 Model B  
- **Input Sensor**: USB Microphone  
- **Output**: LCD Display / Terminal Log  
- **Language**: Python  
- **Audio Processing**: Librosa *(untuk ekstraksi MFCC)*  
- **Machine Learning**: Scikit-learn  
- **Models Tested**: Random Forest, SVM, KNN  

## 📁 Pipeline Pemrosesan Data
1. **Audio Acquisition & Pre-processing**  
   Sistem merekam audio lingkungan dan mengukur level volume. Data hanya akan diproses lebih lanjut jika volume melebihi ambang batas (*threshold*) tertentu untuk menyaring noise latar belakang.

2. **Feature Extraction (MFCC)**  
   Menggunakan **MFCC (Mel-Frequency Cepstral Coefficients)** untuk mengambil fitur unik dari sinyal suara kendaraan. Fitur ini merepresentasikan karakteristik timbral yang membedakan suara mesin motor, mobil, dan truk.

3. **Classification Output**  
   Sistem memberikan output berupa kelas prediksi dan volume suara.

   **Contoh Log Output:**
   ~~~txt
   Volume: 0.4403 | Prediksi: >>> 1 <<<
   Volume: 0.5120 | Prediksi: >>> 2 <<<
   ~~~

   **Keterangan label:**
   - `1 = Motor`
   - `2 = Mobil`
   - `3 = Truk`

## 🛡️ Performa & Evaluasi Model
Berdasarkan pengujian komparatif terhadap tiga algoritma, **Random Forest** dipilih sebagai model utama karena performanya yang paling stabil dan akurat pada Raspberry Pi.

| Model Algorithm | Akurasi | Keterangan |
|---|---:|---|
| Random Forest | 96% | Performa paling unggul dan stabil |
| SVM (Support Vector Machine) | 92% | Cukup baik, namun variabilitas lebih tinggi dari RF |
| KNN (K-Nearest Neighbors) | 85% | Akurasi terendah, kurang ideal untuk presisi tinggi |

## 🔍 Troubleshooting
| Masalah | Penyebab | Solusi |
|---|---|---|
| Low Accuracy | Noise lingkungan tinggi | Sesuaikan threshold volume atau gunakan windshield pada mic |
| Input Lag | CPU Throttling | Pastikan pendinginan Raspberry Pi memadai |
| Microphone Error | Device path berubah | Cek ID perangkat audio dengan `arecord -l` |
| Dependencies Error | Library version mismatch | Pastikan versi `scikit-learn` dan `librosa` sesuai |

## 📝 Catatan Penting
- **Posisi Mikrofon**: Penempatan mikrofon sangat mempengaruhi kualitas MFCC yang diekstraksi. Disarankan mengarah langsung ke sumber suara (jalan raya).
- **Dataset**: Model dilatih menggunakan data suara kendaraan spesifik. Untuk lingkungan baru, disarankan melakukan retraining dengan sampel data setempat.
- **Volume Threshold**: Nilai ambang batas volume pada kode program perlu dikalibrasi sesuai tingkat kebisingan lokasi pemasangan.

> ⚠️ **Disclaimer**: Proyek ini dikembangkan sebagai purwarupa untuk penelitian akademis dan mungkin memerlukan penyesuaian lebih lanjut untuk penggunaan industri skala besar.

