"""
Machine Learning Model for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

class StockPredictor:
    """ML-based stock price predictor using scikit-learn"""
    
    def __init__(self, model_type='random_forest', sequence_length=60, feature_columns=None):
        """
        Initialize the predictor
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'linear'
            sequence_length: Number of past days to use for prediction
            feature_columns: List of feature column names
        """
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or ['open', 'high', 'low', 'close', 'volume']
        self.scaler = MinMaxScaler()
        self.model_type = model_type
        self.model = None
        
        # Initialize the model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Initialized {model_type} predictor")
    
    def create_sequences(self, data):
        """
        Create sequences for training
        
        Args:
            data: Normalized data array
            
        Returns:
            X, y arrays
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            # Flatten the sequence into a single feature vector
            sequence = data[i-self.sequence_length:i].flatten()
            X.append(sequence)
            y.append(data[i, 3])  # Predict close price (index 3)
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, train_ratio=0.8):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with stock data
            train_ratio: Ratio of data to use for training
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        # Select features
        data = df[self.feature_columns].values
        
        # Normalize
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Split into train and validation
        train_size = int(len(X) * train_ratio)
        
        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]
        
        return X_train, X_val, y_train, y_val
    
    def train(self, df, train_ratio=0.8, verbose=True):
        """
        Train the model
        
        Args:
            df: DataFrame with stock data
            train_ratio: Ratio of data to use for training
            verbose: Whether to print training info
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(df, train_ratio)
        
        if verbose:
            print(f"Training {self.model_type} model...")
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            print("-" * 50)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions on both sets
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        metrics = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
        
        if verbose:
            print(f"Training MSE: {train_mse:.6f}")
            print(f"Validation MSE: {val_mse:.6f}")
            print(f"Training MAE: {train_mae:.6f}")
            print(f"Validation MAE: {val_mae:.6f}")
            print(f"Training R²: {train_r2:.4f}")
            print(f"Validation R²: {val_r2:.4f}")
            print("-" * 50)
            print("Training completed!")
        
        return metrics
    
    def predict(self, df, days_ahead=7):
        """
        Make predictions for future days
        
        Args:
            df: DataFrame with stock data
            days_ahead: Number of days to predict
            
        Returns:
            Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        data = df[self.feature_columns].values
        scaled_data = self.scaler.transform(data)
        
        # Get last sequence
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Flatten sequence for prediction
            X = current_sequence.flatten().reshape(1, -1)
            
            # Predict
            pred_value = self.model.predict(X)[0]
            predictions.append(pred_value)
            
            # Update sequence
            new_row = current_sequence[-1].copy()
            new_row[3] = pred_value  # Update close price
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Denormalize predictions
        predictions = np.array(predictions).reshape(-1, 1)
        dummy = np.zeros((len(predictions), len(self.feature_columns)))
        dummy[:, 3] = predictions[:, 0]
        predictions_denorm = self.scaler.inverse_transform(dummy)[:, 3]
        
        return predictions_denorm
    
    def predict_with_confidence(self, df, days_ahead=7):
        """
        Make predictions with confidence intervals (for Random Forest only)
        
        Args:
            df: DataFrame with stock data
            days_ahead: Number of days to predict
            
        Returns:
            Dictionary with predictions, lower_bound, upper_bound
        """
        if self.model_type != 'random_forest':
            predictions = self.predict(df, days_ahead)
            return {
                'predictions': predictions,
                'lower_bound': predictions * 0.95,
                'upper_bound': predictions * 1.05
            }
        
        # For Random Forest, use tree predictions to estimate confidence
        data = df[self.feature_columns].values
        scaled_data = self.scaler.transform(data)
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        lower_bounds = []
        upper_bounds = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            X = current_sequence.flatten().reshape(1, -1)
            
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X)[0] for tree in self.model.estimators_])
            
            # Calculate mean and std
            pred_mean = tree_predictions.mean()
            pred_std = tree_predictions.std()
            
            predictions.append(pred_mean)
            lower_bounds.append(pred_mean - 1.96 * pred_std)  # 95% confidence
            upper_bounds.append(pred_mean + 1.96 * pred_std)
            
            # Update sequence
            new_row = current_sequence[-1].copy()
            new_row[3] = pred_mean
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Denormalize all predictions
        def denormalize(values):
            values = np.array(values).reshape(-1, 1)
            dummy = np.zeros((len(values), len(self.feature_columns)))
            dummy[:, 3] = values[:, 0]
            return self.scaler.inverse_transform(dummy)[:, 3]
        
        return {
            'predictions': denormalize(predictions),
            'lower_bound': denormalize(lower_bounds),
            'upper_bound': denormalize(upper_bounds)
        }
    
    def get_feature_importance(self):
        """Get feature importance (for tree-based models only)"""
        if self.model_type in ['random_forest', 'gradient_boosting']:
            # Calculate importance for each feature in the flattened sequence
            importances = self.model.feature_importances_
            
            # Aggregate by original feature
            n_features = len(self.feature_columns)
            aggregated = np.zeros(n_features)
            
            for i in range(len(importances)):
                feature_idx = i % n_features
                aggregated[feature_idx] += importances[i]
            
            # Normalize
            aggregated = aggregated / aggregated.sum()
            
            return dict(zip(self.feature_columns, aggregated))
        else:
            return None
    
    def save_model(self, filepath):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.sequence_length = model_data['sequence_length']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(end='2024-01-01', periods=500, freq='D')
    np.random.seed(42)
    base_price = 150
    prices = base_price + np.cumsum(np.random.randn(500) * 2)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(500) * 0.5,
        'high': prices + abs(np.random.randn(500) * 1),
        'low': prices - abs(np.random.randn(500) * 1),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 500)
    })
    
    print("Sample data:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    
    # Test all model types
    for model_type in ['random_forest', 'gradient_boosting', 'linear']:
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Initialize predictor
        predictor = StockPredictor(model_type=model_type, sequence_length=60)
        
        # Train model
        metrics = predictor.train(df, train_ratio=0.8)
        
        # Make predictions
        predictions = predictor.predict(df, days_ahead=7)
        
        print(f"\nPredictions for next 7 days:")
        current_price = df['close'].iloc[-1]
        for i, pred in enumerate(predictions, 1):
            change = ((pred - current_price) / current_price) * 100
            print(f"Day {i}: ${pred:.2f} ({change:+.2f}%)")
        
        # Get feature importance (if available)
        importance = predictor.get_feature_importance()
        if importance:
            print("\nFeature Importance:")
            for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {imp:.4f}")
    
    # Save best model (Random Forest)
    print("\n" + "="*60)
    predictor = StockPredictor(model_type='random_forest', sequence_length=60)
    predictor.train(df, train_ratio=0.8)
    predictor.save_model("stock_model.pkl")
    
    # Test predictions with confidence intervals
    print("\nPredictions with confidence intervals:")
    conf_pred = predictor.predict_with_confidence(df, days_ahead=7)
    
    for i in range(7):
        print(f"Day {i+1}: ${conf_pred['predictions'][i]:.2f} "
              f"[${conf_pred['lower_bound'][i]:.2f} - ${conf_pred['upper_bound'][i]:.2f}]")
    
    # Visualize predictions
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices
    plt.plot(df['date'].iloc[-90:], df['close'].iloc[-90:], 
             label='Historical Price', color='blue', linewidth=2)
    
    # Plot predictions
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='D')
    
    plt.plot(future_dates, conf_pred['predictions'], 
             label='Predicted Price', color='red', linewidth=2, linestyle='--')
    
    # Plot confidence interval
    plt.fill_between(future_dates, conf_pred['lower_bound'], conf_pred['upper_bound'], 
                     alpha=0.3, color='red', label='95% Confidence Interval')
    
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.title('Stock Price Prediction with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=150)
    print("\nVisualization saved to prediction_visualization.png")