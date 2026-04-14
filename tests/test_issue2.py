import unittest
import os
import pandas as pd
import tempfile
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        """Buat sample data untuk testing"""
        self.classification_data = pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
            'feature2': [3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        self.regression_data = pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
            'feature2': [3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3],
            'target': [10.5, 15.2, 20.1, 25.8, 30.5, 35.2, 40.1, 45.8, 50.5, 55.2]
        })
        
        # Buat file temporary
        self.classification_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.classification_data.to_csv(self.classification_file.name, index=False)
        self.classification_file.close()
        
        self.regression_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.regression_data.to_csv(self.regression_file.name, index=False)
        self.regression_file.close()
    
    def tearDown(self):
        """Hapus file temporary"""
        os.unlink(self.classification_file.name)
        os.unlink(self.regression_file.name)
    
    def test_load_and_split_classification(self):
        """Test loading classification data"""
        result = DataLoader.load_and_split(self.classification_file.name)
        
        X_train, X_test, y_train, y_test, problem_type = result
        
        # Cek tipe problem
        self.assertEqual(problem_type, 'classification')
        
        # Cek jumlah data
        self.assertEqual(len(X_train) + len(X_test), 10)
        self.assertEqual(len(y_train) + len(y_test), 10)
        
        # Cek stratify: proporsi label harus mirip
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        self.assertAlmostEqual(train_ratio, test_ratio, delta=0.3)
    
    def test_load_and_split_regression(self):
        """Test loading regression data"""
        result = DataLoader.load_and_split(self.regression_file.name)
        
        X_train, X_test, y_train, y_test, problem_type = result
        
        # Cek tipe problem
        self.assertEqual(problem_type, 'regression')
        
        # Cek jumlah data
        self.assertEqual(len(X_train) + len(X_test), 10)
    
    def test_missing_value_raises_error(self):
        """Test bahwa missing value memicu error"""
        # Buat data dengan missing value
        data_with_na = pd.DataFrame({
            'feature1': [1.2, None, 3.4],
            'label': [0, 1, 0]
        })
        
        na_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data_with_na.to_csv(na_file.name, index=False)
        na_file.close()
        
        with self.assertRaises(ValueError) as context:
            DataLoader.load_and_split(na_file.name)
        
        self.assertIn("missing value", str(context.exception).lower())
        
        os.unlink(na_file.name)
    
    def test_target_column_not_found_raises_error(self):
        """Test bahwa target column yang tidak ditemukan memicu error"""
        with self.assertRaises(ValueError) as context:
            DataLoader.load_and_split(self.classification_file.name, target_column='tidak_ada')
        
        self.assertIn("tidak ditemukan", str(context.exception))
    
    def test_auto_detect_target_column(self):
        """Test bahwa target column otomatis menggunakan kolom terakhir"""
        result = DataLoader.load_and_split(self.classification_file.name, target_column=None)
        X_train, X_test, y_train, y_test, problem_type = result
        
        # Kolom terakhir adalah 'label', seharusnya jadi target
        self.assertEqual(problem_type, 'classification')

if __name__ == '__main__':
    unittest.main()
