# Blueprint – ML Training dari Data User

## Tujuan Utama
Membangun sistem ML yang bisa:
1. Menerima data dari user (file CSV/Excel)
2. Melakukan training model (supervised learning – klasifikasi/regresi)
3. Menyimpan model hasil training
4. Menerima data baru untuk prediksi

## Batasan (Anti-Over Engineering)
- Tidak pakai neural network / deep learning (kecuali diminta)
- Tidak pakai MLOps / pipeline kompleks
- Tidak pakai distributed training
- Framework: **scikit-learn + pandas + joblib**
- Deployment: local script + CLI

## Komponen Utama
1. `data_loader.py` – load & validasi data
2. `trainer.py` – training & save model
3. `predictor.py` – load model & prediksi
4. `config.yaml` – konfigurasi target column, test size, dll
5. `main.py` – entry point CLI

## Struktur Folder Final
```
ml_simple_trainer/
├── data/
│   ├── raw/           # Data mentah dari user
│   └── processed/     # Data setelah preprocessing
├── models/            # Model tersimpan (.pkl)
├── src/
│   ├── data_loader.py
│   ├── trainer.py
│   ├── predictor.py
│   └── utils.py
├── config.yaml
├── main.py
└── requirements.txt
```

## Alur Kerja yang Diharapkan
1. User taruh CSV di `data/raw/` 
2. User jalankan `python main.py --mode train --data data/raw/data.csv` 
3. Sistem membaca konfigurasi dari `config.yaml` 
4. Sistem load data, split train/test, deteksi problem type
5. Sistem training model (random forest / logistic regression)
6. Sistem simpan model ke `models/model.pkl` 
7. User jalankan `python main.py --mode predict --input data_baru.csv` 
8. Sistem load model dan output prediksi

## Konfigurasi (config.yaml)
```yaml
target_column: null  # null = kolom terakhir
test_size: 0.2
random_state: 42
model_type: "random_forest"  # atau "logistic_regression"
```

## Catatan Penting untuk Developer
- Hanya untuk supervised learning (klasifikasi/regresi)
- Otomatis deteksi tipe masalah dari kolom target
- Data harus sudah bersih (tidak ada missing value) – user wajib prep dulu
- Tidak perlu tambahan fitur preprocessing
- Tidak perlu hyperparameter tuning
- Tidak perlu API atau web interface
