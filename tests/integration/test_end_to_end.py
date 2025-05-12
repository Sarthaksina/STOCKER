"""
Integration tests for end-to-end prediction workflow.

This module tests the complete prediction pipeline from data ingestion to prediction output.
"""
import os
import tempfile
import shutil
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.entity.config_entity import StockerConfig
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.ml import EnsembleModel, XGBoostModel, LSTMModel, LightGBMModel
from tests.utils import (
    generate_synthetic_price_data,
    generate_synthetic_features,
    assert_predictions_valid,
    assert_metrics_valid,
)


class TestEndToEndPrediction:
    """Test the end-to-end prediction workflow."""

    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        config = StockerConfig()
        # Override config with test values
        config.logs_dir = os.path.join(tempfile.gettempdir(), "stocker_test_logs")
        config.prediction_output_path = os.path.join(tempfile.gettempdir(), "test_predictions.csv")
        config.model_path = os.path.join(tempfile.gettempdir(), "test_models")
        return config

    @pytest.fixture
    def test_data(self):
        """Generate test data for prediction."""
        # Generate synthetic price data
        price_data = generate_synthetic_price_data(n_samples=100)
        # Generate synthetic features
        features = generate_synthetic_features(price_data, n_features=10)
        # Combine into a single DataFrame
        data = pd.concat([price_data, features], axis=1)
        return data

    @pytest.fixture
    def test_model_ensemble(self, test_config):
        """Create a test ensemble model."""
        # Create individual models
        lstm_model = LSTMModel(
            name="test_lstm",
            config={
                "sequence_length": 10,
                "hidden_dim": 32,
                "num_layers": 1,
                "learning_rate": 0.01,
                "epochs": 2,  # Small number for testing
                "batch_size": 16,
                "input_dim": 5,  # Example input dimension
            },
        )

        xgb_model = XGBoostModel(
            name="test_xgb",
            config={
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 10,  # Small number for testing
                "objective": "reg:squarederror",
                "sequence_length": 10,
            },
        )

        lgb_model = LightGBMModel(
            name="test_lgb",
            config={
                "learning_rate": 0.1,
                "num_leaves": 15,
                "n_estimators": 10,  # Small number for testing
                "objective": "regression",
                "sequence_length": 10,
            },
        )

        # Create an ensemble model
        ensemble = EnsembleModel(
            name="test_ensemble",
            models=[lstm_model, xgb_model, lgb_model],
            weights={"test_lstm": 0.3, "test_xgb": 0.4, "test_lgb": 0.3},
            ensemble_strategy="weighted_average",
        )

        return ensemble

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "predictions"), exist_ok=True)
        yield temp_dir
        # Clean up after test
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    def test_prediction_with_preloaded_model(self, test_config, test_data, test_model_ensemble):
        """Test prediction pipeline with a preloaded model."""
        # Create a prediction pipeline
        pipeline = PredictionPipeline(test_config)

        # Manually set the model in the pipeline artifacts
        pipeline.artifacts["model"] = test_model_ensemble
        pipeline.artifacts["model_version"] = "test_version"
        pipeline.artifacts["model_metadata"] = {
            "created_at": datetime.now().isoformat(),
            "model_type": "ensemble",
            "features": test_data.columns.tolist(),
        }

        # Run the pipeline with the test data
        pipeline.prepare_features(input_data=test_data)
        pipeline.run_prediction()

        # Validate the results
        predictions = pipeline.artifacts["predictions"]
        assert_predictions_valid(predictions)
        assert len(predictions) == len(test_data) - test_model_ensemble.config.get(
            "sequence_length", 10
        )

    def test_end_to_end_pipeline(self, test_config, test_data, test_model_ensemble, temp_dir):
        """Test the complete end-to-end prediction pipeline."""
        # Update config with temporary directories
        test_config.prediction_output_path = os.path.join(temp_dir, "predictions", "predictions.csv")
        test_config.model_path = os.path.join(temp_dir, "models")
        test_config.logs_dir = os.path.join(temp_dir, "logs")

        # Save test model to the model path
        os.makedirs(test_config.model_path, exist_ok=True)
        model_dir = os.path.join(test_config.model_path, "test_ensemble_v1")
        os.makedirs(model_dir, exist_ok=True)
        test_model_ensemble.save(model_dir)

        # Create metadata file for the model
        metadata = {
            "model_id": "test_ensemble_v1",
            "model_type": "ensemble",
            "created_at": datetime.now().isoformat(),
            "features": test_data.columns.tolist(),
            "version": "v1",
            "is_latest": True,
        }
        import json
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Create a prediction pipeline
        pipeline = PredictionPipeline(test_config)

        # Run the full pipeline with model_id and test data
        results = pipeline.run(input_data=test_data, model_id="test_ensemble_v1")

        # Validate the results
        assert "predictions" in results
        assert "confidence" in results
        assert "model_version" in results
        assert "performance_metrics" in results

        predictions = results["predictions"]
        assert_predictions_valid(predictions)

        # Check that the output file was created
        assert os.path.exists(test_config.prediction_output_path)
        output_df = pd.read_csv(test_config.prediction_output_path)
        assert len(output_df) > 0

        # Check performance metrics
        performance_metrics = results["performance_metrics"]
        assert "total_execution_time" in performance_metrics
        assert performance_metrics["total_execution_time"] > 0

    def test_error_handling(self, test_config):
        """Test error handling in the prediction pipeline."""
        # Create a prediction pipeline
        pipeline = PredictionPipeline(test_config)

        # Test with missing model
        with pytest.raises(Exception):
            pipeline.run_prediction()

        # Test with missing features
        pipeline.artifacts["model"] = object()  # Just a placeholder
        with pytest.raises(Exception):
            pipeline.run_prediction()

        # Test with invalid model_id
        with pytest.raises(Exception):
            pipeline.run(model_id="nonexistent_model") 