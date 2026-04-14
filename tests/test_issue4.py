import unittest
import os
import tempfile
import pandas as pd
import joblib
from src.data_loader import DataLoader
from src.trainer import ModelTrainer
from src.predictor import Predictor

class TestPredictor(unittest.TestCase):
    
    def setUp(self):
        """Setup test data dan model terlatih"""
        # Buat classification data untuk training
        self.classification_data = pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
            'feature2': [3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Buat regression data untuk training
        self.regression_data = pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3, 13.4, 14.5, 15.6],
            'feature2': [3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3, 13.4, 14.5, 15.6, 16.7, 17.8],
            'target': [10.5, 15.2, 20.1, 25.8, 30.5, 35.2, 40.1, 45.8, 50.5, 55.2, 60.1, 65.8, 70.5, 75.2, 80.1]
        })
        
        # Buat temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Simpan data ke file
        self.classification_file = os.path.join(self.temp_dir.name, 'classification.csv')
        self.classification_data.to_csv(self.classification_file, index=False)
        
        self.regression_file = os.path.join(self.temp_dir.name, 'regression.csv')
        self.regression_data.to_csv(self.regression_file, index=False)
        
        # Train classification model
        X_train, X_test, y_train, y_test, problem_type = DataLoader.load_and_split(
            self.classification_file
        )
        self.class_trainer = ModelTrainer(model_type='random_forest', problem_type='classification')
        self.class_trainer.train(X_train, y_train)
        
        # Save classification model
        self.class_model_path = os.path.join(self.temp_dir.name, 'class_model.pkl')
        self.class_trainer.save(self.class_model_path)
        
        # Train regression model
        X_train, X_test, y_train, y_test, problem_type = DataLoader.load_and_split(
            self.regression_file
        )
        self.reg_trainer = ModelTrainer(problem_type='regression')
        self.reg_trainer.train(X_train, y_train)
        
        # Save regression model
        self.reg_model_path = os.path.join(self.temp_dir.name, 'reg_model.pkl')
        self.reg_trainer.save(self.reg_model_path)
        
        # Buat data baru untuk prediksi (feature only, tanpa label)
        self.new_classification_data = pd.DataFrame({
            'feature1': [2.5, 7.5, 9.5],
            'feature2': [4.5, 9.5, 11.5]
        })
        
        self.new_regression_data = pd.DataFrame({
            'feature1': [3.5, 8.5, 13.5],
            'feature2': [5.5, 10.5, 15.5]
        })
        
        # Simpan data baru ke file
        self.new_class_file = os.path.join(self.temp_dir.name, 'new_classification.csv')
        self.new_classification_data.to_csv(self.new_class_file, index=False)
        
        self.new_reg_file = os.path.join(self.temp_dir.name, 'new_regression.csv')
        self.new_regression_data.to_csv(self.new_reg_file, index=False)
        
        # Buat data dengan missing value untuk test error
        self.data_with_na = pd.DataFrame({
            'feature1': [1.2, None, 3.4],
            'feature2': [2.3, 4.5, 6.7]
        })
        self.na_file = os.path.join(self.temp_dir.name, 'data_with_na.csv')
        self.data_with_na.to_csv(self.na_file, index=False)
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_load_model_classification(self):
        """Test loading classification model"""
        model = Predictor.load_model(self.class_model_path)
        
        self.assertIsNotNone(model)
        
        # Model harus bisa predict
        predictions = model.predict(self.new_classification_data)
        self.assertEqual(len(predictions), 3)
    
    def test_load_model_regression(self):
        """Test loading regression model"""
        model = Predictor.load_model(self.reg_model_path)
        
        self.assertIsNotNone(model)
        
        # Model harus bisa predict
        predictions = model.predict(self.new_regression_data)
        self.assertEqual(len(predictions), 3)
    
    def test_predict_classification(self):
        """Test prediction with classification model"""
        model = Predictor.load_model(self.class_model_path)
        predictions = Predictor.predict(model, self.new_class_file)
        
        self.assertEqual(len(predictions), 3)
        # Predictions should be 0 or 1 for classification
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_predict_regression(self):
        """Test prediction with regression model"""
        model = Predictor.load_model(self.reg_model_path)
        predictions = Predictor.predict(model, self.new_reg_file)
        
        self.assertEqual(len(predictions), 3)
        # Predictions should be float numbers
        self.assertTrue(all(isinstance(p, float) for p in predictions))
    
    def test_predict_with_missing_value_raises_error(self):
        """Test that missing value in prediction data raises error"""
        model = Predictor.load_model(self.class_model_path)
        
        with self.assertRaises(ValueError) as context:
            Predictor.predict(model, self.na_file)
        
        self.assertIn("missing value", str(context.exception).lower())
    
    def test_predict_with_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises error"""
        model = Predictor.load_model(self.class_model_path)
        
        with self.assertRaises(ValueError) as context:
            Predictor.predict(model, "file_yang_tidak_ada.csv")
        
        self.assertIn("tidak ditemukan", str(context.exception).lower())
    
    def test_load_model_nonexistent_path_raises_error(self):
        """Test that loading nonexistent model raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            Predictor.load_model("model_yang_tidak_ada.pkl")
    
    def test_predict_with_empty_path_raises_error(self):
        """Test that empty file path raises error"""
        model = Predictor.load_model(self.class_model_path)
        
        with self.assertRaises(ValueError) as context:
            Predictor.predict(model, "")
        
        self.assertIn("new_data_path", str(context.exception).lower())
    
    def test_predict_with_none_path_raises_error(self):
        """Test that None file path raises error"""
        model = Predictor.load_model(self.class_model_path)
        
        with self.assertRaises(ValueError) as context:
            Predictor.predict(model, None)
        
        self.assertIn("new_data_path", str(context.exception).lower())

if __name__ == '__main__':
    unittest.main()
