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
        
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_packages = ['pandas', 'scikit-learn', 'joblib', 'pyyaml']
        for pkg in required_packages:
            with self.subTest(package=pkg):
                self.assertIn(pkg, content, f"{pkg} tidak ditemukan di requirements.txt")
    
    def test_config_exists_and_valid(self):
        """Test config.yaml ada dan valid YAML"""
        self.assertTrue(os.path.exists('config.yaml'))
        
        with open('config.yaml', 'r', encoding='utf-8') as f:
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
    
    def test_config_values_are_valid(self):
        """Test nilai config dalam range yang valid"""
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validasi test_size antara 0 dan 1
        self.assertGreaterEqual(config['test_size'], 0)
        self.assertLessEqual(config['test_size'], 1)
        
        # Validasi random_state harus integer positif
        self.assertIsInstance(config['random_state'], int)
        self.assertGreaterEqual(config['random_state'], 0)
        
        # Validasi model_type (case insensitive)
        allowed_models = ['random_forest', 'logistic_regression']
        self.assertIn(config['model_type'].lower(), allowed_models)
    
    def test_setup_py_exists_and_valid(self):
        """Test setup.py ada dan valid Python syntax"""
        self.assertTrue(os.path.exists('setup.py'))
        
        import ast
        with open('setup.py', 'r', encoding='utf-8') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            self.fail(f"setup.py syntax error: {e}")
    
    def test_gitignore_exists_and_has_patterns(self):
        """Test .gitignore ada dan berisi pattern penting"""
        self.assertTrue(os.path.exists('.gitignore'))
        
        with open('.gitignore', 'r', encoding='utf-8') as f:
            content = f.read()
        
        important_patterns = ['__pycache__', '*.pkl', '*.csv', '.env', 'dist/']
        for pattern in important_patterns:
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, content, f"Pattern {pattern} tidak ditemukan di .gitignore")

if __name__ == '__main__':
    unittest.main()
