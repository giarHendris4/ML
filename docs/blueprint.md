# Blueprint – ML Training dari Data User

## Tujuan Utama
Membangun sistem ML yang bisa:
1. Menerima data dari user (file CSV/Excel)
2. Melakukan training model (supervised learning – klasifikasi/regresi)
3. Menyimpan model hasil training
4. Menerima data baru untuk prediksi

## Batasan (Anti-Over Engineering)
- Tidak pakai neural network / deep learning
- Tidak pakai MLOps / pipeline kompleks
- Framework: scikit-learn + pandas + joblib
- Deployment: local script + CLI

## Komponen Utama
1. data_loader.py – load & validasi data
2. trainer.py – training & save model
3. predictor.py – load model & prediksi
4. config.yaml – konfigurasi
5. main.py – entry point CLI

## Struktur Folder
ml_simple_trainer/
├── data/raw/
├── data/processed/
├── models/
├── src/
└── tests/
