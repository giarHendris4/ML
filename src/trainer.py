import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from pathlib import Path

class ModelTrainer:
    def __init__(self, model_type='random_forest', problem_type='classification'):
        """
        Initialize trainer with model configuration.
        
        Args:
            model_type: 'random_forest' or 'logistic_regression'
            problem_type: 'classification' or 'regression'
        """
        self.model_type = model_type
        self.problem_type = problem_type
        self.model = self._init_model()
    
    def _init_model(self):
        """Initialize model based on configuration."""
        if self.problem_type == 'classification':
            if self.model_type == 'logistic_regression':
                return LogisticRegression(max_iter=1000, random_state=42)
            else:
                return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train(self, X_train, y_train):
        """Train the model on training data."""
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate model and return metrics."""
        predictions = self.model.predict(X_test)
        
        if self.problem_type == 'classification':
            score = accuracy_score(y_test, predictions)
            metric_name = 'accuracy'
        else:
            score = mean_squared_error(y_test, predictions)
            metric_name = 'mse'
        
        return {metric_name: score}
    
    def save(self, path='models/model.pkl'):
        """Save model to disk using joblib."""
        Path('models').mkdir(exist_ok=True)
        joblib.dump(self.model, path)
        return path
