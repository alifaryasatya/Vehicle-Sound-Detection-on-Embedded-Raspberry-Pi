# Vehicle Sound Detection on Embedded Raspberry Pi

![Python](https://img.shields.io/badge/Language-Python%203-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red)
![ML](https://img.shields.io/badge/AI-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

![Device Setup](assets/device_setup.jpg)
*(Ganti dengan foto perangkat Raspberry Pi Anda)*

## 📌 Overview
Sistem deteksi dan klasifikasi suara kendaraan berbasis *Embedded Machine Learning* yang dirancang untuk beroperasi pada Raspberry Pi. Sistem ini memanfaatkan ekstraksi fitur audio **MFCC (Mel-Frequency Cepstral Coefficients)** untuk membedakan jenis kendaraan (Mobil, Motor, Truk) secara *real-time* berdasarkan karakteristik akustik mesinnya.

Berbeda dengan sistem berbasis visi komputer, pendekatan berbasis audio ini lebih efisien secara komputasi dan tetap efektif dalam kondisi pencahayaan rendah atau malam hari.

## 🎯 Tujuan Proyek
- **Real-time Monitoring**: Mengklasifikasikan jenis kendaraan yang lewat secara langsung di perangkat *edge*.
- **High Accuracy**: Mencapai akurasi klasifikasi tinggi menggunakan algoritma *Machine Learning* yang teruji.
- **Embedded Implementation**: Mengoptimalkan pemrosesan sinyal suara agar berjalan lancar pada perangkat dengan sumber daya terbatas (Raspberry Pi).

## 📊 Diagram Alur Sistem
```mermaid
graph LR
    A[Mic USB] -->|Raw Audio| B[Pre-processing]
    B -->|Ekstraksi MFCC| C{ML Model}
    C -->|Klasifikasi| D[Output LCD/Log]
    
    subgraph "Machine Learning Core"
    C
    end