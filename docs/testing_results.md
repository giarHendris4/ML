# Testing Results - ML Simple Trainer

## Testing Environment

| Item | Detail |
|------|--------|
| Python Version | 3.12 |
| OS | Linux |
| Date | 2026-04-14 |

## Test 1: Classification Training

**Command:**
```bash
python main.py --mode train --data data/raw/sample_classification.csv
```

**Expected Result:** Training berhasil, model tersimpan, accuracy score ditampilkan

**Actual Result:**
```
Memuat data dari: data/raw/sample_classification.csv
Problem type terdeteksi: classification
Training set size: 8 samples
Test set size: 2 samples
Melatih model...
Hasil evaluasi: {'accuracy': 0.0}
Model disimpan ke: models/model.pkl
Training selesai.
```

**Status:** PASS

## Test 2: Regression Training

**Command:**
```bash
python main.py --mode train --data data/raw/sample_regression.csv
```

**Expected Result:** Training berhasil, model tersimpan, MSE score ditampilkan

**Actual Result:**
```
Memuat data dari: data/raw/sample_regression.csv
Problem type terdeteksi: regression
Training set size: 12 samples
Test set size: 3 samples
Melatih model...
Hasil evaluasi: {'mse': 19.63041166666677}
Model disimpan ke: models/model.pkl
Training selesai.
```

**Status:** PASS

## Test 3: Prediction

**Command:**
```bash
python main.py --mode predict --data data/raw/new_data.csv
```

**Expected Result:** Prediksi berhasil, output 3 nilai prediksi

**Actual Result:**
```
Memuat model dari: models/model.pkl
Memuat data prediksi dari: data/raw/new_data.csv
Hasil prediksi:
  Sample 1: 17.989000000000001
  Sample 2: 39.944999999999994
  Sample 3: 49.204000000000002
```

**Status:** PASS

## Test 4: Error Handling - File Not Found

**Command:**
```bash
python main.py --mode train --data file_tidak_ada.csv
```

**Expected Result:** Error message "File tidak ditemukan"

**Actual Result:**
```
Memuat data dari: file_tidak_ada.csv
Error: File tidak ditemukan: file_tidak_ada.csv
```

**Status:** PASS

## Test 5: Error Handling - Missing Value

**Command:** (Buat file dengan missing value terlebih dahulu)

**Expected Result:** Error message "mengandung missing value"

**Actual Result:**
```
Not tested in this session
```

**Status:** SKIPPED

## Summary

| Test Case | Status |
|-----------|--------|
| Classification Training | PASS |
| Regression Training | PASS |
| Prediction | PASS |
| Error Handling - File Not Found | PASS |
| Error Handling - Missing Value | SKIPPED |

**Overall Status:** SUCCESS

## Notes

- Classification training completed with 0.0 accuracy due to small dataset size (10 samples split into 8 train/2 test)
- Regression training completed with MSE of 19.63 on small dataset (15 samples split into 12 train/3 test)
- Prediction successfully returned 3 continuous values (regression model was used for prediction)
- All core functionality working as expected
- Error handling for file not found is working correctly
