# Project Issues

## Issue 1: Inisialisasi Struktur Proyek & Dependencies

### Task List:
- [ ] Buat folder `data/raw/` 
- [ ] Buat folder `data/processed/` 
- [ ] Buat folder `models/` 
- [ ] Buat folder `src/` 
- [ ] Buat folder `notebooks/` (opsional)
- [ ] Buat file `requirements.txt` dengan isi:
  ```
  pandas==2.2.0
  scikit-learn==1.4.0
  joblib==1.3.2
  pyyaml==6.0.1
  ```
- [ ] Buat file `config.yaml` dengan isi:
  ```yaml
  target_column: null
  test_size: 0.2
  random_state: 42
  model_type: "random_forest"
  ```

## Issue 2: Implementasi Data Loader
### Task List:
- [ ] Buat file `src/data_loader.py`
- [ ] Implementasi fungsi load_data()
- [ ] Implementasi validasi data
- [ ] Implementasi split train/test
- [ ] Buat unit test

## Issue 3: Implementasi Trainer
### Task List:
- [ ] Buat file `src/trainer.py`
- [ ] Implementasi training model
- [ ] Implementasi save model
- [ ] Buat unit test

## Issue 4: Implementasi Predictor
### Task List:
- [ ] Buat file `src/predictor.py`
- [ ] Implementasi load model
- [ ] Implementasi prediksi
- [ ] Buat unit test

## Issue 5: Implementasi Main CLI
### Task List:
- [ ] Buat file `main.py`
- [ ] Implementasi CLI interface
- [ ] Integrasi semua komponen
- [ ] Buat integration test

## Issue 6: Testing & Documentation
### Task List:
- [ ] Buat sample data
- [ ] Jalankan end-to-end test
- [ ] Update README.md
- [ ] Buat user guide
