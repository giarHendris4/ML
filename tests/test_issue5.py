import unittest
import subprocess
import os
import tempfile
import pandas as pd
import yaml
import sys

class TestMainCLI(unittest.TestCase):
    
    def setUp(self):
        """Setup test data dan konfigurasi"""
        # Buat temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Buat sample data untuk training
        self.training_data = pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
            'feature2': [3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        self.training_file = os.path.join(self.temp_dir.name, 'train.csv')
        self.training_data.to_csv(self.training_file, index=False)
        
        # Buat sample data untuk prediksi
        self.prediction_data = pd.DataFrame({
            'feature1': [2.5, 7.5, 9.5],
            'feature2': [4.5, 9.5, 11.5]
        })
        self.prediction_file = os.path.join(self.temp_dir.name, 'predict.csv')
        self.prediction_data.to_csv(self.prediction_file, index=False)
        
        # Buat config sementara
        self.config = {
            'target_column': 'label',
            'test_size': 0.2,
            'random_state': 42,
            'model_type': 'random_forest'
        }
        self.config_file = os.path.join(self.temp_dir.name, 'test_config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_train_mode_success(self):
        """Test training mode berhasil"""
        result = subprocess.run([
            sys.executable, 'main.py',
            '--mode', 'train',
            '--data', self.training_file,
            '--config', self.config_file
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        # Status code harus 0 (success)
        self.assertEqual(result.returncode, 0)
        
        # Output harus mengandung indikator sukses
        self.assertIn("Training selesai", result.stdout)
        self.assertIn("Problem type terdeteksi", result.stdout)
    
    def test_predict_mode_success(self):
        """Test prediction mode berhasil setelah training"""
        # Training dulu
        subprocess.run([
            sys.executable, 'main.py',
            '--mode', 'train',
            '--data', self.training_file,
            '--config', self.config_file
        ], capture_output=True, text=True)
        
        # Kemudian predict
        result = subprocess.run([
            sys.executable, 'main.py',
            '--mode', 'predict',
            '--data', self.prediction_file,
            '--config', self.config_file
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Hasil prediksi", result.stdout)
    
    def test_train_mode_missing_data_file(self):
        """Test training dengan file data tidak ada"""
        result = subprocess.run([
            sys.executable, 'main.py',
            '--mode', 'train',
            '--data', 'file_yang_tidak_ada.csv',
            '--config', self.config_file
        ], capture_output=True, text=True)
        
        # Status code harus 1 (error)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Error", result.stdout)
    
    def test_predict_mode_missing_model(self):
        """Test prediction tanpa model terlatih"""
        # Hapus model jika ada
        model_path = 'models/model.pkl'
        if os.path.exists(model_path):
            os.remove(model_path)
        
        result = subprocess.run([
            sys.executable, 'main.py',
            '--mode', 'predict',
            '--data', self.prediction_file,
            '--config', self.config_file
        ], capture_output=True, text=True)
        
        # Status code harus 1 (error)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Error", result.stdout)
    
    def test_invalid_mode(self):
        """Test dengan mode yang tidak valid"""
        result = subprocess.run([
            sys.executable, 'main.py',
            '--mode', 'invalid_mode',
            '--data', self.training_file
        ], capture_output=True, text=True)
        
        # argparse akan menangani, return code non-zero
        self.assertNotEqual(result.returncode, 0)
    
    def test_missing_required_argument(self):
        """Test tanpa argument --data"""
        result = subprocess.run([
            sys.executable, 'main.py',
            '--mode', 'train'
        ], capture_output=True, text=True)
        
        # argparse akan menangani, return code non-zero
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr.lower())

if __name__ == '__main__':
    unittest.main()
