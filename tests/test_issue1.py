import unittest
import os
import yaml

class TestIssue1(unittest.TestCase):
    
    def test_folders_exist(self):
        """Test semua folder yang diperlukan ada"""
        folders = ['data/raw', 'data/processed', 'models', 'src', 'notebooks']
        for folder in folders:
            with self.subTest(folder=folder):
                self.assertTrue(os.path.exists(folder), f"Folder {folder} tidak ditemukan")
                self.assertTrue(os.path.isdir(folder), f"{folder} bukan folder")
    
    def test_requirements_exists(self):
        """Test requirements.txt ada dan berisi library yang benar"""
        self.assertTrue(os.path.exists('requirements.txt'))
        
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        required_packages = ['pandas', 'scikit-learn', 'joblib', 'pyyaml']
        for pkg in required_packages:
            with self.subTest(package=pkg):
                self.assertIn(pkg, content, f"{pkg} tidak ditemukan di requirements.txt")
    
    def test_config_exists_and_valid(self):
        """Test config.yaml ada dan valid YAML"""
        self.assertTrue(os.path.exists('config.yaml'))
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test required keys
        self.assertIn('target_column', config)
        self.assertIn('test_size', config)
        self.assertIn('random_state', config)
        self.assertIn('model_type', config)
        
        # Test default values
        self.assertIsNone(config['target_column'])
        self.assertEqual(config['test_size'], 0.2)
        self.assertEqual(config['random_state'], 42)
        self.assertIn(config['model_type'], ['random_forest', 'logistic_regression'])

if __name__ == '__main__':
    unittest.main()
