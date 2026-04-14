import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    @staticmethod
    def load_and_split(file_path, target_column=None, test_size=0.2, random_state=42):
        """
        Load CSV, split features & target, lalu split train/test.
        
        Returns:
            X_train, X_test, y_train, y_test, problem_type
        
        Raises:
            ValueError: Jika data mengandung missing value atau target column tidak ditemukan
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError(f"file_path harus berupa string yang valid, received: {file_path}")
        
        if not os.path.exists(file_path):
            raise ValueError(f"File tidak ditemukan: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validasi: cek missing value
        if df.isnull().sum().sum() > 0:
            raise ValueError("Data mengandung missing value. Harap bersihkan dulu.")
        
        # Tentukan target column
        if target_column is None:
            target_column = df.columns[-1]
        elif target_column not in df.columns:
            raise ValueError(f"Kolom {target_column} tidak ditemukan.")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Deteksi problem type yang lebih akurat
        if y.dtype == 'object':
            problem_type = 'classification'
        elif y.dtype in ['int64', 'float64']:
            unique_ratio = len(y.unique()) / len(y)
            if unique_ratio < 0.05 or len(y.unique()) < 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            problem_type = 'classification'
        
        # Split data
        stratify_param = y if problem_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        return X_train, X_test, y_train, y_test, problem_type
