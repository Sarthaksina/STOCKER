# STOCKER Pro Machine Learning Module

This module contains the enhanced machine learning architecture for STOCKER Pro, featuring model ensembles that combine multiple prediction approaches for superior performance.

## Model Architecture

STOCKER Pro uses a modular approach with the following key components:

1. **Base Model Interface**: All models implement a common interface (`BaseModel`) with standardized methods for training, prediction, and evaluation.

2. **Individual Models**:
   - **LSTM**: Deep learning model specialized for time series data with memory capabilities.
   - **XGBoost**: Gradient boosting model optimized for structured data with excellent feature importance capabilities.
   - **LightGBM**: Efficient gradient boosting model with leaf-wise tree growth, ideal for large feature spaces.

3. **Ensemble Model**: Combines predictions from multiple base models using various strategies:
   - **Weighted Average**: Simple weighted combination of model predictions
   - **Voting**: Direction-based voting for classification tasks (trend prediction)
   - **Stacking**: Meta-model trained on base model predictions

## Usage Examples

### Basic Model Training and Prediction

```python
from src.ml import LSTMModel, XGBoostModel, LightGBMModel

# Configure LSTM model
lstm_config = {
    "input_dim": 1,
    "hidden_dim": 64,
    "num_layers": 2,
    "sequence_length": 10,
    "epochs": 50
}
lstm_model = LSTMModel(name="stock_lstm", config=lstm_config)

# Train model
lstm_model.fit(train_data, train_data)  # For time series, X and y are often the same

# Make predictions
predictions = lstm_model.predict(test_data)

# Evaluate model
metrics = lstm_model.evaluate(test_data, test_targets)
print(f"MSE: {metrics['mse']}, Dir. Accuracy: {metrics['directional_accuracy']}")

# Save model
lstm_model.save("models/lstm_stock_predictor")

# Load model
loaded_model = LSTMModel(name="loaded_lstm")
loaded_model.load("models/lstm_stock_predictor")
```

### Creating an Ensemble Model

```python
from src.ml import LSTMModel, XGBoostModel, LightGBMModel, EnsembleModel

# Create individual models
lstm_model = LSTMModel(name="lstm", config=lstm_config)
xgb_model = XGBoostModel(name="xgb", config=xgb_config)
lgb_model = LightGBMModel(name="lgb", config=lgb_config)

# Train individual models
for model in [lstm_model, xgb_model, lgb_model]:
    model.fit(train_data, train_data)

# Create ensemble with custom weights
ensemble_config = {
    "ensemble_method": "weighted_avg",
    "weights": {
        "lstm": 0.4,  # Higher weight for LSTM
        "xgb": 0.3,
        "lgb": 0.3
    }
}
ensemble = EnsembleModel(
    name="stock_ensemble", 
    config=ensemble_config, 
    models=[lstm_model, xgb_model, lgb_model]
)

# Make predictions with ensemble
ensemble_predictions = ensemble.predict(test_data)

# Save ensemble
ensemble.save("models/stock_ensemble")
```

### Using Stacking Ensemble

```python
# Create stacking ensemble
stacking_config = {
    "ensemble_method": "stacking",
    "meta_model_type": "lightgbm",  # Meta-model type
    "meta_model_config": {          # Meta-model config
        "n_estimators": 100,
        "learning_rate": 0.05
    },
    "cv_folds": 5,                 # Cross-validation folds for meta-features
    "stack_with_orig_features": True  # Include original features with meta-features
}

stacking_ensemble = EnsembleModel(
    name="stacking_ensemble", 
    config=stacking_config, 
    models=[lstm_model, xgb_model, lgb_model]
)

# Train stacking ensemble (this trains the meta-model)
stacking_ensemble.fit(train_data, train_data)
```

## Financial Evaluation Metrics

STOCKER Pro includes specialized metrics for financial models:

- **Directional Accuracy**: Percentage of correct trend predictions (up/down)
- **Weighted Directional Accuracy**: Accuracy weighted by magnitude of price movements
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return divided by maximum drawdown
- **Trading Strategy Returns**: Simulated returns from using model predictions

You can evaluate models using these metrics:

```python
from src.ml.evaluation import evaluate_financial_model, plot_financial_evaluation

# Comprehensive evaluation
metrics = evaluate_financial_model(
    y_true=test_prices,
    y_pred=predictions,
    include_strategy_simulation=True,
    transaction_cost=0.001
)

# Visualize results
fig, axes = plot_financial_evaluation(
    y_true=test_prices,
    y_pred=predictions,
    dates=test_dates
)
```

## Cloud Training Integration

STOCKER Pro integrates with ThunderCompute for resource-intensive model training:

```python
from src.cloud_training.thunder_compute import ThunderComputeClient

# Initialize client
tc_client = ThunderComputeClient()

# Upload training data
tc_client.upload_data("data/stock_data.csv", "training/stock_data.csv")

# Submit training job
job_id = tc_client.submit_job(
    job_name="xgboost_stock_predictor",
    model_type="xgboost",
    data_path="s3://thundercompute-stocker-pro/training/stock_data.csv",
    config=xgb_config,
    instance_type="ml.g4dn.xlarge"  # GPU instance
)

# Wait for job completion
tc_client.wait_for_job(job_id)

# Load trained model
from src.ml import XGBoostModel
model = tc_client.load_trained_model(job_id, XGBoostModel)
```

## Advanced Features

- **Feature Importance Analysis**: XGBoost and LightGBM models provide feature importance metrics to understand prediction drivers
- **Confidence Intervals**: `predict_proba()` method provides prediction intervals for regression tasks
- **Transfer Learning**: Models can be fine-tuned on new data after initial training
- **Model Versioning**: Saving and loading includes model metadata for versioning

For complete API documentation, see the individual model class documentation. 