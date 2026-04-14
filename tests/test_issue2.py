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
        
        # Buat temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Buat file CSV dalam temporary directory
        self.classification_file_path = os.path.join(self.temp_dir.name, 'classification.csv')
        self.classification_data.to_csv(self.classification_file_path, index=False)
        
        self.regression_file_path = os.path.join(self.temp_dir.name, 'regression.csv')
        self.regression_data.to_csv(self.regression_file_path, index=False)
    
    def tearDown(self):
        """Hapus temporary directory"""
        self.temp_dir.cleanup()
    
    def test_load_and_split_classification(self):
        """Test loading classification data"""
        result = DataLoader.load_and_split(self.classification_file_path)
        
        X_train, X_test, y_train, y_test, problem_type = result
        
        # Cek tipe problem
        self.assertEqual(problem_type, 'classification')
        
        # Cek jumlah data
        self.assertEqual(len(X_train) + len(X_test), 10)
        self.assertEqual(len(y_train) + len(y_test), 10)
        
        # Cek stratify: proporsi label harus mirip
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        self.assertAlmostEqual(train_ratio, test_ratio, delta=0.2)
    
    def test_load_and_split_regression(self):
        """Test loading regression data"""
        result = DataLoader.load_and_split(self.regression_file_path)
        
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
            DataLoader.load_and_split(self.classification_file_path, target_column='tidak_ada')
        
        self.assertIn("tidak ditemukan", str(context.exception))
    
    def test_auto_detect_target_column(self):
        """Test bahwa target column otomatis menggunakan kolom terakhir"""
        result = DataLoader.load_and_split(self.classification_file_path, target_column=None)
        X_train, X_test, y_train, y_test, problem_type = result
        
        # Kolom terakhir adalah 'label', seharusnya jadi target
        self.assertEqual(problem_type, 'classification')

    def test_empty_file_path_raises_error(self):
        """Test bahwa file path kosong memicu error"""
        with self.assertRaises(ValueError) as context:
            DataLoader.load_and_split("")
        
        self.assertIn("file_path", str(context.exception).lower())
    
    def test_none_file_path_raises_error(self):
        """Test bahwa file path None memicu error"""
        with self.assertRaises(ValueError) as context:
            DataLoader.load_and_split(None)
        
        self.assertIn("file_path", str(context.exception).lower())
    
    def test_file_not_found_raises_error(self):
        """Test bahwa file tidak ditemukan memicu error"""
        with self.assertRaises(ValueError) as context:
            DataLoader.load_and_split("file_yang_tidak_ada.csv")
        
        self.assertIn("tidak ditemukan", str(context.exception))

if __name__ == '__main__':
    unittest.main()
