import unittest
import os
import tempfile
import pandas as pd
import joblib
from src.data_loader import DataLoader
from src.trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    
    def setUp(self):
        """Setup test data menggunakan DataLoader dari Issue 2"""
        # Buat classification data
        self.classification_data = pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
            'feature2': [3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Buat regression data
        self.regression_data = pd.DataFrame({
            'feature1': [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3, 13.4, 14.5, 15.6],
            'feature2': [3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1, 11.2, 12.3, 13.4, 14.5, 15.6, 16.7, 17.8],
            'target': [10.5, 15.2, 20.1, 25.8, 30.5, 35.2, 40.1, 45.8, 50.5, 55.2, 60.1, 65.8, 70.5, 75.2, 80.1]
        })
        
        # Buat temporary file untuk data
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.classification_file = os.path.join(self.temp_dir.name, 'classification.csv')
        self.classification_data.to_csv(self.classification_file, index=False)
        
        self.regression_file = os.path.join(self.temp_dir.name, 'regression.csv')
        self.regression_data.to_csv(self.regression_file, index=False)
        
        # Load dan split data menggunakan DataLoader
        X_train, X_test, y_train, y_test, self.class_problem_type = DataLoader.load_and_split(
            self.classification_file
        )
        self.class_X_train = X_train
        self.class_X_test = X_test
        self.class_y_train = y_train
        self.class_y_test = y_test
        
        X_train, X_test, y_train, y_test, self.reg_problem_type = DataLoader.load_and_split(
            self.regression_file
        )
        self.reg_X_train = X_train
        self.reg_X_test = X_test
        self.reg_y_train = y_train
        self.reg_y_test = y_test
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_init_classification_random_forest(self):
        """Test initialization for classification with random forest"""
        trainer = ModelTrainer(model_type='random_forest', problem_type='classification')
        
        self.assertEqual(trainer.model_type, 'random_forest')
        self.assertEqual(trainer.problem_type, 'classification')
        self.assertIsNotNone(trainer.model)
    
    def test_init_classification_logistic_regression(self):
        """Test initialization for classification with logistic regression"""
        trainer = ModelTrainer(model_type='logistic_regression', problem_type='classification')
        
        self.assertEqual(trainer.model_type, 'logistic_regression')
        self.assertEqual(trainer.problem_type, 'classification')
        self.assertIsNotNone(trainer.model)
    
    def test_init_regression(self):
        """Test initialization for regression"""
        trainer = ModelTrainer(problem_type='regression')
        
        self.assertEqual(trainer.problem_type, 'regression')
        self.assertIsNotNone(trainer.model)
    
    def test_train_classification(self):
        """Test training for classification model"""
        trainer = ModelTrainer(model_type='random_forest', problem_type='classification')
        trainer.train(self.class_X_train, self.class_y_train)
        
        # Model should have been trained
        self.assertIsNotNone(trainer.model)
        
        # Should be able to predict
        predictions = trainer.model.predict(self.class_X_test)
        self.assertEqual(len(predictions), len(self.class_y_test))
    
    def test_train_regression(self):
        """Test training for regression model"""
        trainer = ModelTrainer(problem_type='regression')
        trainer.train(self.reg_X_train, self.reg_y_train)
        
        # Model should have been trained
        self.assertIsNotNone(trainer.model)
        
        # Should be able to predict
        predictions = trainer.model.predict(self.reg_X_test)
        self.assertEqual(len(predictions), len(self.reg_y_test))
    
    def test_evaluate_classification(self):
        """Test evaluation for classification model"""
        trainer = ModelTrainer(model_type='random_forest', problem_type='classification')
        trainer.train(self.class_X_train, self.class_y_train)
        
        metrics = trainer.evaluate(self.class_X_test, self.class_y_test)
        
        self.assertIn('accuracy', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_evaluate_regression(self):
        """Test evaluation for regression model"""
        trainer = ModelTrainer(problem_type='regression')
        trainer.train(self.reg_X_train, self.reg_y_train)
        
        metrics = trainer.evaluate(self.reg_X_test, self.reg_y_test)
        
        self.assertIn('mse', metrics)
        self.assertGreaterEqual(metrics['mse'], 0)
    
    def test_save_model(self):
        """Test saving model to disk"""
        trainer = ModelTrainer(model_type='random_forest', problem_type='classification')
        trainer.train(self.class_X_train, self.class_y_train)
        
        # Save to temporary location
        temp_model_path = os.path.join(self.temp_dir.name, 'test_model.pkl')
        saved_path = trainer.save(temp_model_path)
        
        self.assertEqual(saved_path, temp_model_path)
        self.assertTrue(os.path.exists(temp_model_path))
        
        # Verify model can be loaded
        loaded_model = joblib.load(temp_model_path)
        self.assertIsNotNone(loaded_model)
    
    def test_save_auto_creates_models_directory(self):
        """Test that save method auto-creates models directory"""
        trainer = ModelTrainer(model_type='random_forest', problem_type='classification')
        trainer.train(self.class_X_train, self.class_y_train)
        
        # Save with default path
        trainer.save()
        
        self.assertTrue(os.path.exists('models'))
        self.assertTrue(os.path.exists('models/model.pkl'))

if __name__ == '__main__':
    unittest.main()
