import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

class PitcherStrikeoutPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.scaler = None
        
    def load_data(self, data_dir='data'):
        """Load all pitcher data from CSV files"""
        all_data = []
        
        # Load data from all available years
        for filename in os.listdir(data_dir):
            if filename.startswith('pitcher_training_data_') and filename.endswith('.csv'):
                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath)
                all_data.append(df)
        
        if not all_data:
            raise ValueError("No data files found in the data directory")
        
        # Combine all data
        self.data = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(self.data)} pitcher records")
        
    def preprocess_data(self):
        """Prepare features for the model"""
        # Select only age, ERA, and WHIP as features
        feature_cols = ['age', 'era', 'whip']
        
        # Check which columns exist in our data
        available_cols = [col for col in feature_cols if col in self.data.columns]
        print(f"Using features: {available_cols}")
        
        # Remove rows with missing values in our target or key features
        self.data_clean = self.data.dropna(subset=['strikeouts'] + available_cols)
        
        # Calculate strikeouts per game
        self.data_clean['strikeouts_per_game'] = self.data_clean['strikeouts'] / self.data_clean['games']
        
        # Remove any infinite values from division
        self.data_clean = self.data_clean.replace([np.inf, -np.inf], np.nan)
        self.data_clean = self.data_clean.dropna(subset=['strikeouts_per_game'])
        
        # Prepare features and target
        X = self.data_clean[available_cols]
        y = self.data_clean['strikeouts_per_game']
        
        # Remove any remaining infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_columns = available_cols
        self.X = X
        self.y = y
        
        print(f"Preprocessed data: {len(self.X)} samples, {len(self.feature_columns)} features")
        print(f"Target: strikeouts per game (range: {y.min():.2f} - {y.max():.2f})")
        
        # Show feature statistics
        print(f"\nFeature Statistics:")
        for col in available_cols:
            print(f"  {col}: {X[col].mean():.2f} (avg), {X[col].min():.2f} (min), {X[col].max():.2f} (max)")
        
    def train_model(self, test_size=0.2, random_state=42):
        """Train the Random Forest model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Create and train the model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Mean Absolute Error: {mae:.2f} strikeouts per game")
        print(f"Root Mean Square Error: {rmse:.2f} strikeouts per game")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        for _, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return mae, rmse, r2
    
    def predict_strikeouts_per_game(self, pitcher_stats):
        """Predict strikeouts per game for a given pitcher"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Create a DataFrame with the pitcher's stats
        input_df = pd.DataFrame([pitcher_stats])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value
        
        # Select only the features used in training
        input_features = input_df[self.feature_columns]
        
        # Make prediction
        prediction = self.model.predict(input_features)[0]
        
        return round(prediction, 2)
    
    def get_sample_pitcher(self):
        """Get a sample pitcher from the data for demonstration"""
        sample = self.data_clean.sample(1).iloc[0]
        pitcher_stats = {}
        
        for col in self.feature_columns:
            pitcher_stats[col] = sample[col]
        
        return pitcher_stats, sample['strikeouts_per_game']

def main():
    """Main function to demonstrate the predictor"""
    print("=== Pitcher Strikeouts Per Game Predictor ===\n")
    
    # Initialize predictor
    predictor = PitcherStrikeoutPredictor()
    
    try:
        # Load and preprocess data
        print("Loading data...")
        predictor.load_data()
        
        print("Preprocessing data...")
        predictor.preprocess_data()
        
        # Train model
        print("Training model...")
        mae, rmse, r2 = predictor.train_model()
        
        # Demonstrate prediction
        print("\n" + "="*50)
        print("DEMONSTRATION: Predicting Strikeouts Per Game")
        print("="*50)
        
        # Get a sample pitcher
        sample_stats, actual_strikeouts = predictor.get_sample_pitcher()
        
        print(f"\nSample Pitcher Stats:")
        for feature, value in sample_stats.items():
            print(f"  {feature}: {value}")
        
        # Make prediction
        predicted_strikeouts = predictor.predict_strikeouts_per_game(sample_stats)
        
        print(f"\nPrediction Results:")
        print(f"  Predicted Strikeouts per Game: {predicted_strikeouts}")
        print(f"  Actual Strikeouts per Game: {actual_strikeouts:.2f}")
        print(f"  Difference: {abs(predicted_strikeouts - actual_strikeouts):.2f}")
        
        # Interactive prediction
        print(f"\n" + "="*50)
        print("INTERACTIVE PREDICTION")
        print("="*50)
        print("Enter pitcher statistics to predict strikeouts per game (press Enter to skip):")
        
        # Get user input for the three key stats
        user_stats = {}
        
        print(f"\nEnter pitcher statistics:")
        print(f"  Age: Pitcher's age (e.g., 25)")
        print(f"  ERA: Earned Run Average (e.g., 3.50)")
        print(f"  WHIP: Walks + Hits per Inning Pitched (e.g., 1.20)")
        
        for feature in predictor.feature_columns:
            try:
                value = input(f"{feature}: ")
                if value.strip():
                    user_stats[feature] = float(value)
                else:
                    print(f"No value entered for {feature}, using average value")
                    user_stats[feature] = predictor.X[feature].mean()
            except ValueError:
                print(f"Invalid input for {feature}, using average value")
                user_stats[feature] = predictor.X[feature].mean()
        
        # Make prediction
        prediction = predictor.predict_strikeouts_per_game(user_stats)
        print(f"\nPredicted Strikeouts per Game: {prediction}")
        
        # Show what the model learned
        print(f"\nModel Insights:")
        print(f"  Based on {len(predictor.X)} pitcher records")
        print(f"  Using only: Age, ERA, and WHIP")
        print(f"  Average strikeouts per game in dataset: {predictor.y.mean():.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the data files are in the 'data' directory.")

if __name__ == "__main__":
    main()
