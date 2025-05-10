# demo_pipeline.py
"""
Demo pipeline for STOCKER Pro.
This script demonstrates how to use the ML pipeline system.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.core.config import StockerConfig
from src.ml.pipelines import TrainingPipeline, PredictionPipeline, train_model
from src.ml.models import LSTMModel, XGBoostModel, EnsembleModel
from src.ml.evaluation import ModelEvaluator, compare_models
from src.data.ingestion import get_stock_data

def demo_training_pipeline():
    """Demonstrate the training pipeline with sample data."""
    print("=== Training Pipeline Demo ===")
    
    # Create a configuration
    config = StockerConfig(
        mode="train",
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2020-01-01",
        end_date="2022-12-31",
        target_col="close",
        model_type="ensemble",
        model_params={
            "base_models": ["xgboost", "lstm"],
            "ensemble_method": "weighted",
            "weights": [0.6, 0.4]
        }
    )
    
    # Simulate getting feature data (would normally come from feature engineering)
    # For demo, we'll create a synthetic dataset
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="B")
    n_samples = len(dates)
    
    # Create synthetic features and target
    np.random.seed(42)
    features = pd.DataFrame({
        "date": dates,
        "open": np.random.normal(100, 10, n_samples),
        "high": np.random.normal(105, 12, n_samples),
        "low": np.random.normal(95, 8, n_samples),
        "volume": np.random.normal(1000000, 200000, n_samples),
        "ma_5": np.random.normal(100, 8, n_samples),
        "ma_10": np.random.normal(100, 7, n_samples),
        "ma_20": np.random.normal(100, 6, n_samples),
        "rsi_14": np.random.normal(50, 15, n_samples),
        "macd": np.random.normal(0, 2, n_samples),
        "bollinger_up": np.random.normal(110, 5, n_samples),
        "bollinger_down": np.random.normal(90, 5, n_samples),
    })
    
    # Create a synthetic close price (target)
    features["close"] = features["open"] + np.random.normal(0, 3, n_samples)
    
    # Add the features to the config
    config.features_df = features
    
    # Run the pipeline
    pipeline = TrainingPipeline(config)
    artifacts = pipeline.run()
    
    print(f"Training completed. Model type: {config.model_type}")
    print(f"Evaluation metrics: {artifacts['evaluation_metrics']}")
    
    return artifacts

def demo_prediction_pipeline(artifacts):
    """Demonstrate the prediction pipeline with a trained model."""
    print("\n=== Prediction Pipeline Demo ===")
    
    # Get the trained model from artifacts
    model = artifacts["model"]
    
    # Create test data (last 30 days from the training data)
    test_data = artifacts["X"].iloc[-30:].copy()
    
    # Create a prediction config
    predict_config = StockerConfig(
        mode="predict",
        model_path=artifacts["model_path"] if "model_path" in artifacts else None,
        features=test_data
    )
    
    # If we don't have model_path, add the model directly
    if "model_path" not in artifacts:
        predict_config.model = model
    
    # Run the prediction pipeline
    pipeline = PredictionPipeline(predict_config)
    prediction_artifacts = pipeline.run()
    
    print(f"Prediction completed.")
    print(f"Generated {len(prediction_artifacts['predictions'])} predictions")
    
    return prediction_artifacts

def demo_model_comparison():
    """Demonstrate model comparison with multiple models."""
    print("\n=== Model Comparison Demo ===")
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Features and target
    X = np.random.normal(0, 1, (n_samples, n_features))
    y = 2 * X[:, 0] + 0.5 * X[:, 1] - 1 * X[:, 2] + np.random.normal(0, 0.5, n_samples)
    
    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    
    # Initialize models
    models = {
        "XGBoost": XGBoostModel(
            name="xgb_demo",
            model_type="xgboost",
            config={"n_estimators": 100, "max_depth": 3}
        ),
        "LSTM": LSTMModel(
            name="lstm_demo",
            model_type="lstm",
            config={"units": 32, "epochs": 10}
        ),
        "Ensemble": EnsembleModel(
            name="ensemble_demo",
            model_type="ensemble",
            config={
                "base_models": ["xgboost", "lstm"],
                "ensemble_method": "weighted"
            }
        )
    }
    
    # Train models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_df, y_series)
    
    # Compare models
    evaluator = ModelEvaluator()
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_df)
        evaluator.evaluate_model(name, y_series.values, y_pred)
    
    # Get comparison DataFrame
    comparison_df = evaluator.compare_models(metrics=["mse", "mae", "r2", "directional_accuracy"])
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Plotting
    try:
        fig = evaluator.plot_comparison(metrics=["mse", "mae", "r2", "directional_accuracy"])
        plt.title("Model Performance Comparison")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
    
    return comparison_df

def main():
    """Run the complete demonstration."""
    print("STOCKER Pro ML Pipeline Demonstration")
    print("=" * 40)
    
    # Run training pipeline demo
    artifacts = demo_training_pipeline()
    
    # Run prediction pipeline demo
    prediction_artifacts = demo_prediction_pipeline(artifacts)
    
    # Run model comparison demo
    comparison_df = demo_model_comparison()
    
    print("\nDemonstration completed successfully.")

if __name__ == "__main__":
    main()
