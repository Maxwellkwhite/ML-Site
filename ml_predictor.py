import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PitcherStrikeoutPredictor:
    """
    Machine Learning model for predicting pitcher strikeouts.
    
    This class handles:
    - Data preprocessing
    - Model training
    - Predictions
    - Model persistence
    - Performance evaluation
    """
    
    def __init__(self, model_path='models/strikeout_predictor.pkl'):
        """
        Initialize the predictor with model path and default parameters.
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.feature_columns = []
        self.target_column = 'strikeouts'
        self.scaler = None
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        logger.info("PitcherStrikeoutPredictor initialized")
    
    def load_data(self, data_source='api'):
        """
        Load training data from various sources.
        
        Args:
            data_source (str): Source of data ('api', 'csv', 'database')
            
        Returns:
            pd.DataFrame: Loaded and preprocessed data
        """
        try:
            if data_source == 'api':
                # TODO: Implement API data loading
                # This could fetch from MLB Stats API, Baseball Reference, etc.
                logger.info("API data loading not yet implemented")
                return pd.DataFrame()
            
            elif data_source == 'csv':
                # Load from CSV file
                csv_path = 'data/pitcher_data.csv'
                if os.path.exists(csv_path):
                    data = pd.read_csv(csv_path)
                    logger.info(f"Loaded {len(data)} records from CSV")
                    return data
                else:
                    logger.warning(f"CSV file not found: {csv_path}")
                    return pd.DataFrame()
            
            else:
                logger.error(f"Unknown data source: {data_source}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, data):
        """
        Preprocess the raw data for training.
        
        Args:
            data (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        try:
            if data.empty:
                logger.warning("No data to preprocess")
                return data
            
            # Make a copy to avoid modifying original data
            processed_data = data.copy()
            
            # Handle missing values
            processed_data = self._handle_missing_values(processed_data)
            
            # Feature engineering
            processed_data = self._engineer_features(processed_data)
            
            # Encode categorical variables
            processed_data = self._encode_categorical(processed_data)
            
            # Scale numerical features
            processed_data = self._scale_features(processed_data)
            
            logger.info(f"Preprocessed data shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset."""
        # Fill numerical columns with median
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        return data
    
    def _engineer_features(self, data):
        """Create new features from existing data."""
        # Example features (adjust based on available data):
        # - Pitcher age
        # - Days rest
        # - Season stats averages
        # - Home/Away indicator
        # - Weather conditions
        # - Opponent strength
        
        # Add placeholder features for now
        if 'age' in data.columns:
            data['age_squared'] = data['age'] ** 2
        
        if 'era' in data.columns:
            data['era_category'] = pd.cut(data['era'], bins=5, labels=['Excellent', 'Good', 'Average', 'Below Average', 'Poor'])
        
        return data
    
    def _encode_categorical(self, data):
        """Encode categorical variables."""
        # One-hot encode categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != self.target_column:  # Don't encode target variable
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
                data.drop(col, axis=1, inplace=True)
        
        return data
    
    def _scale_features(self, data):
        """Scale numerical features."""
        # For now, we'll use simple min-max scaling
        # In production, you might want to use StandardScaler or RobustScaler
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col != self.target_column:  # Don't scale target variable
                if data[col].max() != data[col].min():
                    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        
        return data
    
    def train(self, data, test_size=0.2, random_state=42):
        """
        Train the strikeout prediction model.
        
        Args:
            data (pd.DataFrame): Training data
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training metrics
        """
        try:
            if data.empty:
                logger.error("No data provided for training")
                return {}
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Define features (exclude target column)
            self.feature_columns = [col for col in processed_data.columns if col != self.target_column]
            
            if not self.feature_columns:
                logger.error("No feature columns found")
                return {}
            
            # Prepare X and y
            X = processed_data[self.feature_columns]
            y = processed_data[self.target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Initialize and train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
            
            logger.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            metrics['feature_importance'] = feature_importance
            
            self.is_trained = True
            
            logger.info(f"Training completed. R² Score: {metrics['r2']:.4f}")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {}
    
    def predict(self, pitcher_data):
        """
        Make strikeout predictions for given pitcher data.
        
        Args:
            pitcher_data (dict or pd.DataFrame): Pitcher information
            
        Returns:
            float: Predicted strikeouts
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("Model not trained. Please train the model first.")
                return None
            
            # Convert dict to DataFrame if needed
            if isinstance(pitcher_data, dict):
                pitcher_data = pd.DataFrame([pitcher_data])
            
            # Preprocess the data
            processed_data = self.preprocess_data(pitcher_data)
            
            # Ensure all required features are present
            missing_features = set(self.feature_columns) - set(processed_data.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    processed_data[feature] = 0
            
            # Select only the features used in training
            X = processed_data[self.feature_columns]
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            logger.info(f"Prediction made: {prediction:.2f} strikeouts")
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def save_model(self):
        """Save the trained model to disk."""
        try:
            if self.model is not None:
                model_data = {
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'is_trained': self.is_trained,
                    'timestamp': datetime.now().isoformat()
                }
                
                joblib.dump(model_data, self.model_path)
                logger.info(f"Model saved to {self.model_path}")
                return True
            else:
                logger.warning("No model to save")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load a trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
                self.target_column = model_data['target_column']
                self.is_trained = model_data['is_trained']
                
                logger.info(f"Model loaded from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self):
        """Get feature importance from the trained model."""
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        else:
            logger.warning("No trained model available")
            return {}
    
    def evaluate_model(self, test_data):
        """
        Evaluate the model on new test data.
        
        Args:
            test_data (pd.DataFrame): Test data with actual strikeout values
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            if not self.is_trained:
                logger.error("Model not trained")
                return {}
            
            # Preprocess test data
            processed_data = self.preprocess_data(test_data)
            
            # Prepare features and target
            X_test = processed_data[self.feature_columns]
            y_test = processed_data[self.target_column]
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            logger.info(f"Model evaluation - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Create predictor instance
    predictor = PitcherStrikeoutPredictor()
    
    # Example of how to use the class
    print("Pitcher Strikeout Predictor initialized!")
    print("To use this class:")
    print("1. Load data: predictor.load_data()")
    print("2. Train model: predictor.train(data)")
    print("3. Make predictions: predictor.predict(pitcher_data)")
    print("4. Save model: predictor.save_model()") 