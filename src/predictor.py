import joblib
import pandas as pd

class Predictor:
    @staticmethod
    def load_model(path='models/model.pkl'):
        """
        Load model from disk using joblib.
        
        Args:
            path: Path to the saved model file
            
        Returns:
            Loaded model object
        """
        return joblib.load(path)
    
    @staticmethod
    def predict(model, new_data_path):
        """
        Load new data from CSV and make predictions.
        
        Args:
            model: Trained model object
            new_data_path: Path to CSV file containing new data (features only)
            
        Returns:
            Numpy array of predictions
            
        Raises:
            ValueError: If data contains missing values
        """
        df = pd.read_csv(new_data_path)
        
        # Validasi missing value
        if df.isnull().sum().sum() > 0:
            raise ValueError("Data prediksi mengandung missing value. Harap bersihkan dulu.")
        
        return model.predict(df)
