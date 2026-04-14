# Streamlit Web UI - ML Simple Trainer

## Cara Menjalankan

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Jalankan Streamlit app:
```bash
streamlit run app.py
```

3. Buka browser di http://localhost:8501

## Fitur

### Tab 1: Train Model
- Upload file CSV
- Pilih target column
- Konfigurasi model type dan test size
- Training dengan visualisasi hasil
- Lihat feature importance (untuk Random Forest)

### Tab 2: Make Predictions
- Upload file CSV (feature only)
- Dapatkan prediksi
- Download hasil sebagai CSV
- Visualisasi chart untuk regression

## Catatan

- Model disimpan di `models/model.pkl` 
- Data training harus memiliki kolom target
- Data prediksi harus memiliki kolom fitur yang sama dengan data training
