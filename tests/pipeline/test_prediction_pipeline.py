"""
Tests for the prediction pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.entity.config_entity import StockerConfig
from src.exception.exception import ModelLoadingError, FeatureEngineeringError, StockerPredictionError
from src.configuration.config_validator import ConfigValidationError

class TestPredictionPipeline:
    """Tests for the PredictionPipeline class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = StockerConfig()
        config.logs_dir = "test_logs"
        config.prediction_output_path = "test_predictions.csv"
        return config
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model that returns dummy predictions."""
        model = MagicMock()
        model.predict.return_value = np.array([1.0, 2.0, 3.0])
        model.predict_proba.return_value = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
        return model
    
    @pytest.fixture
    def mock_features(self):
        """Create mock features for testing."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.1, 0.2, 0.3],
            'feature3': [10, 20, 30]
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up
        shutil.rmtree(temp_dir)
    
    def test_init(self, mock_config):
        """Test pipeline initialization."""
        pipeline = PredictionPipeline(mock_config)
        assert pipeline.config == mock_config
        assert isinstance(pipeline.artifacts, dict)
        assert isinstance(pipeline.performance_metrics, dict)
        assert pipeline.start_time > 0
    
    @patch('src.pipeline.prediction_pipeline.validate_prediction_config')
    def test_validate_config_success(self, mock_validate, mock_config):
        """Test successful config validation."""
        # Setup the mock
        mock_validate.return_value = True
        
        # Create pipeline and call validate_config
        pipeline = PredictionPipeline(mock_config)
        pipeline.validate_config()
        
        # Assert the mock was called
        mock_validate.assert_called_once()
        
        # Assert metrics were tracked
        assert 'config_validation_time' in pipeline.performance_metrics
    
    @patch('src.pipeline.prediction_pipeline.validate_prediction_config')
    def test_validate_config_failure(self, mock_validate, mock_config):
        """Test config validation failure."""
        # Setup the mock to raise an exception
        mock_validate.side_effect = ConfigValidationError("Invalid config")
        
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        
        # Assert that calling validate_config raises the expected exception
        with pytest.raises(ConfigValidationError):
            pipeline.validate_config()
    
    @patch('src.pipeline.prediction_pipeline.load_latest_model')
    def test_load_model_success(self, mock_load_model, mock_config, mock_model):
        """Test successful model loading."""
        # Setup the mock
        mock_load_model.return_value = (mock_model, "v1.0", {"model_type": "test"})
        
        # Create pipeline and call load_model
        pipeline = PredictionPipeline(mock_config)
        pipeline.load_model()
        
        # Assert the mock was called
        mock_load_model.assert_called_once()
        
        # Assert artifacts were stored
        assert pipeline.artifacts['model'] == mock_model
        assert pipeline.artifacts['model_version'] == "v1.0"
        assert pipeline.artifacts['model_metadata'] == {"model_type": "test"}
        
        # Assert metrics were tracked
        assert 'model_loading_time' in pipeline.performance_metrics
    
    @patch('src.pipeline.prediction_pipeline.load_latest_model')
    def test_load_model_failure(self, mock_load_model, mock_config):
        """Test model loading failure."""
        # Setup the mock to raise an exception
        mock_load_model.side_effect = Exception("Model loading failed")
        
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        
        # Assert that calling load_model raises the expected exception
        with pytest.raises(ModelLoadingError):
            pipeline.load_model()
    
    @patch('src.pipeline.prediction_pipeline.feature_engineer_for_prediction')
    def test_prepare_features_success(self, mock_feature_engineer, mock_config, mock_features):
        """Test successful feature preparation."""
        # Setup the mock
        mock_feature_engineer.return_value = mock_features
        
        # Create pipeline and call prepare_features
        pipeline = PredictionPipeline(mock_config)
        pipeline.prepare_features()
        
        # Assert the mock was called
        mock_feature_engineer.assert_called_once()
        
        # Assert artifacts were stored
        assert pipeline.artifacts['features'].equals(mock_features)
        
        # Assert metrics were tracked
        assert 'feature_engineering_time' in pipeline.performance_metrics
    
    @patch('src.pipeline.prediction_pipeline.feature_engineer_for_prediction')
    def test_prepare_features_with_input_data(self, mock_feature_engineer, mock_config, mock_features):
        """Test feature preparation with provided input data."""
        # Setup the mock
        mock_feature_engineer.return_value = mock_features
        input_data = pd.DataFrame({'raw_feature': [1, 2, 3]})
        
        # Create pipeline and call prepare_features with input_data
        pipeline = PredictionPipeline(mock_config)
        pipeline.prepare_features(input_data=input_data)
        
        # Assert the mock was called with the input_data
        mock_feature_engineer.assert_called_once()
        # Check that the second argument of the first call is input_data
        args, kwargs = mock_feature_engineer.call_args
        assert 'input_data' in kwargs
        assert kwargs['input_data'].equals(input_data)
        
        # Assert artifacts were stored
        assert pipeline.artifacts['features'].equals(mock_features)
    
    @patch('src.pipeline.prediction_pipeline.feature_engineer_for_prediction')
    def test_prepare_features_failure(self, mock_feature_engineer, mock_config):
        """Test feature preparation failure."""
        # Setup the mock to raise an exception
        mock_feature_engineer.side_effect = Exception("Feature engineering failed")
        
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        
        # Assert that calling prepare_features raises the expected exception
        with pytest.raises(FeatureEngineeringError):
            pipeline.prepare_features()
    
    def test_validate_features(self, mock_config):
        """Test feature validation."""
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        
        # Create features with issues
        features = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan],  # Missing value
            'feature2': [0.1, np.inf, 0.3],  # Infinite value
            'feature3': [10, 20, 30]
        })
        
        # Call _validate_features
        pipeline._validate_features(features)
        
        # No assertion needed as this is just testing that the method runs without errors
        # The actual warnings are logged, but we're not testing the logging here
    
    @patch('src.pipeline.prediction_pipeline.predict')
    def test_run_prediction_success(self, mock_predict, mock_config, mock_model, mock_features):
        """Test successful prediction."""
        # Setup the mock
        predictions = np.array([1.0, 2.0, 3.0])
        confidence = np.array([0.8, 0.9, 0.7])
        mock_predict.return_value = (predictions, confidence)
        
        # Create pipeline and prepare artifacts
        pipeline = PredictionPipeline(mock_config)
        pipeline.artifacts['model'] = mock_model
        pipeline.artifacts['features'] = mock_features
        
        # Call run_prediction
        pipeline.run_prediction()
        
        # Assert the mock was called
        mock_predict.assert_called_once_with(mock_model, mock_features, mock_config)
        
        # Assert artifacts were stored
        np.testing.assert_array_equal(pipeline.artifacts['predictions'], predictions)
        np.testing.assert_array_equal(pipeline.artifacts['confidence'], confidence)
        
        # Assert metrics were tracked
        assert 'prediction_time' in pipeline.performance_metrics
    
    def test_run_prediction_missing_model(self, mock_config):
        """Test prediction without loading a model first."""
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        
        # Assert that calling run_prediction without a model raises the expected exception
        with pytest.raises(StockerPredictionError, match="Model not loaded"):
            pipeline.run_prediction()
    
    def test_run_prediction_missing_features(self, mock_config, mock_model):
        """Test prediction without preparing features first."""
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        pipeline.artifacts['model'] = mock_model
        
        # Assert that calling run_prediction without features raises the expected exception
        with pytest.raises(StockerPredictionError, match="Features not prepared"):
            pipeline.run_prediction()
    
    @patch('src.pipeline.prediction_pipeline.predict')
    def test_run_prediction_failure(self, mock_predict, mock_config, mock_model, mock_features):
        """Test prediction failure."""
        # Setup the mock to raise an exception
        mock_predict.side_effect = Exception("Prediction failed")
        
        # Create pipeline and prepare artifacts
        pipeline = PredictionPipeline(mock_config)
        pipeline.artifacts['model'] = mock_model
        pipeline.artifacts['features'] = mock_features
        
        # Assert that calling run_prediction raises the expected exception
        with pytest.raises(StockerPredictionError):
            pipeline.run_prediction()
    
    @patch('src.pipeline.prediction_pipeline.save_predictions')
    def test_save_success(self, mock_save, mock_config, temp_dir):
        """Test successful saving of predictions."""
        # Setup the mock
        save_path = os.path.join(temp_dir, "predictions.csv")
        mock_save.return_value = save_path
        
        # Create pipeline and prepare artifacts
        pipeline = PredictionPipeline(mock_config)
        pipeline.artifacts['predictions'] = np.array([1.0, 2.0, 3.0])
        pipeline.artifacts['confidence'] = np.array([0.8, 0.9, 0.7])
        pipeline.artifacts['model_metadata'] = {"model_type": "test"}
        
        # Call save
        result_path = pipeline.save(output_path=save_path)
        
        # Assert the mock was called
        mock_save.assert_called_once()
        
        # Assert the returned path is correct
        assert result_path == save_path
        
        # Assert metrics were tracked
        assert 'save_time' in pipeline.performance_metrics
    
    def test_save_missing_predictions(self, mock_config):
        """Test saving without making predictions first."""
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        
        # Assert that calling save without predictions raises the expected exception
        with pytest.raises(StockerPredictionError, match="No predictions to save"):
            pipeline.save()
    
    @patch('src.pipeline.prediction_pipeline.save_predictions')
    def test_save_failure(self, mock_save, mock_config):
        """Test saving failure."""
        # Setup the mock to raise an exception
        mock_save.side_effect = Exception("Saving failed")
        
        # Create pipeline and prepare artifacts
        pipeline = PredictionPipeline(mock_config)
        pipeline.artifacts['predictions'] = np.array([1.0, 2.0, 3.0])
        
        # Assert that calling save raises the expected exception
        with pytest.raises(StockerPredictionError):
            pipeline.save()
    
    @patch('src.pipeline.prediction_pipeline.PredictionPipeline.validate_config')
    @patch('src.pipeline.prediction_pipeline.PredictionPipeline.load_model')
    @patch('src.pipeline.prediction_pipeline.PredictionPipeline.prepare_features')
    @patch('src.pipeline.prediction_pipeline.PredictionPipeline.run_prediction')
    @patch('src.pipeline.prediction_pipeline.PredictionPipeline.save')
    def test_run_full_pipeline(self, mock_save, mock_run_prediction, 
                               mock_prepare_features, mock_load_model, 
                               mock_validate_config, mock_config):
        """Test running the full pipeline."""
        # Setup mocks
        mock_save.return_value = "test_predictions.csv"
        
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        
        # Mock artifacts
        pipeline.artifacts = {
            'predictions': np.array([1.0, 2.0, 3.0]),
            'confidence': np.array([0.8, 0.9, 0.7]),
            'model_version': "v1.0"
        }
        
        # Call run
        result = pipeline.run()
        
        # Assert all pipeline steps were called
        mock_validate_config.assert_called_once()
        mock_load_model.assert_called_once()
        mock_prepare_features.assert_called_once()
        mock_run_prediction.assert_called_once()
        mock_save.assert_called_once()
        
        # Assert the result contains the expected keys
        assert 'predictions' in result
        assert 'confidence' in result
        assert 'model_version' in result
        assert 'performance_metrics' in result
        
        # Assert total execution time was tracked
        assert 'total_execution_time' in result['performance_metrics']
    
    @patch('src.pipeline.prediction_pipeline.PredictionPipeline.validate_config')
    def test_run_pipeline_failure(self, mock_validate_config, mock_config):
        """Test pipeline failure at one of the steps."""
        # Setup the mock to raise an exception
        mock_validate_config.side_effect = ConfigValidationError("Invalid config")
        
        # Create pipeline
        pipeline = PredictionPipeline(mock_config)
        
        # Assert that calling run raises the expected exception
        with pytest.raises(ConfigValidationError):
            pipeline.run() 