# ThunderCompute Setup Guide for FinPulse Project

This guide details how to set up and use ThunderCompute's free tier for training resource-intensive machine learning models without requiring local high-performance hardware.

## 1. Account Setup

### Registration
1. Go to [ThunderCompute's website](https://thundercompute.com)
2. Sign up for a free account using your email
3. Verify your email address
4. Complete your profile and select the "Free Tier" option

### Access Configuration
1. Navigate to the API section in your dashboard
2. Create a new API key with read/write permissions
3. Store this API key securely in your project's environment variables
4. Download the ThunderCompute CLI tool:
```bash
pip install thundercompute-cli
```
5. Configure CLI with your API key:
```bash
thundercompute configure --api-key YOUR_API_KEY
```

## 2. Project Configuration

### Directory Structure
Create the following structure in your project:
```
finpulse/
├── thundercompute/
│   ├── train_scripts/
│   │   ├── train_xgboost.py
│   │   ├── train_lightgbm.py
│   │   └── train_lstm.py
│   ├── config/
│   │   ├── xgboost_params.json
│   │   ├── lightgbm_params.json
│   │   └── lstm_params.json
│   ├── utils/
│   │   ├── data_prep.py
│   │   └── model_utils.py
│   ├── requirements-thunder.txt
│   └── run.py
```

### Requirements File
Create a `requirements-thunder.txt` file with all necessary dependencies:
```
numpy==1.21.6
pandas==1.3.5
scikit-learn==1.0.2
xgboost==1.6.2
lightgbm==3.3.3
tensorflow==2.9.0
keras==2.9.0
matplotlib==3.5.2
seaborn==0.11.2
joblib==1.1.0
ray[tune]==2.0.0
shap==0.41.0
optuna==3.0.3
```

### Configuration Files
Create JSON configuration files for each model. Example for XGBoost:
```json
{
    "model_params": {
        "objective": "reg:squarederror",
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 1000,
        "early_stopping_rounds": 50
    },
    "training_params": {
        "test_size": 0.2,
        "cv_folds": 5,
        "random_state": 42
    },
    "hyperopt_params": {
        "n_trials": 20,
        "timeout": 3600
    }
}
```

## 3. Data Preparation & Upload

### Data Preparation Script
Create a script to prepare and upload your data:

```python
# thundercompute/utils/data_prep.py
import pandas as pd
import numpy as np
from thundercompute_cli import upload_dataset

def prepare_and_upload_data(raw_data_path, processed_data_path, remote_path):
    """
    Prepare and upload data to ThunderCompute
    """
    # Load data
    df = pd.read_csv(raw_data_path)
    
    # Preprocess data
    # ... (your preprocessing steps)
    
    # Save processed data locally
    df.to_csv(processed_data_path, index=False)
    
    # Upload to ThunderCompute
    dataset_id = upload_dataset(
        local_path=processed_data_path,
        remote_path=remote_path,
        description="Processed financial data for model training"
    )
    
    print(f"Data uploaded successfully with ID: {dataset_id}")
    return dataset_id
```

### Execute Data Upload
```bash
python -c "from thundercompute.utils.data_prep import prepare_and_upload_data; prepare_and_upload_data('data/raw/stock_data.csv', 'data/processed/stock_data_processed.csv', 'finpulse/data/stock_data_processed.csv')"
```

## 4. Model Training Scripts

### XGBoost Training Script
```python
# thundercompute/train_scripts/train_xgboost.py
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import optuna
import joblib
from thundercompute_cli import download_dataset, upload_model

def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': 1000,
        'early_stopping_rounds': 50
    }
    
    model = xgb.XGBRegressor(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    preds = model.predict(X_val)
    rmse = np.sqrt(np.mean((preds - y_val) ** 2))
    return rmse

def train_xgboost(config_path, data_id, output_path):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Download dataset
    local_data_path = download_dataset(data_id)
    data = pd.read_csv(local_data_path)
    
    # Prepare features and target
    X = data.drop(['target', 'date'], axis=1, errors='ignore')
    y = data['target']
    
    # Train/validation split
    train_size = int(len(X) * (1 - config['training_params']['test_size']))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=config['hyperopt_params']['n_trials'],
        timeout=config['hyperopt_params']['timeout']
    )
    
    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'objective': 'reg:squarederror',
        'n_estimators': 1000,
        'early_stopping_rounds': 50
    })
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Save model locally
    model_path = f"{output_path}/xgboost_model.joblib"
    joblib.dump(model, model_path)
    
    # Upload model to ThunderCompute
    model_id = upload_model(
        local_path=model_path,
        remote_path=f"finpulse/models/xgboost_model.joblib",
        description="XGBoost model for stock prediction"
    )
    
    print(f"Model trained and uploaded with ID: {model_id}")
    return model_id

if __name__ == "__main__":
    train_xgboost(
        config_path="thundercompute/config/xgboost_params.json", 
        data_id="YOUR_DATA_ID_HERE",
        output_path="models"
    )
```

### Similar scripts should be created for LightGBM and LSTM models

## 5. Main Run Script

```python
# thundercompute/run.py
import argparse
import os
import time
from thundercompute_cli import create_job, get_job_status, download_model

def run_training_job(model_type, data_id):
    """
    Submit a training job to ThunderCompute
    """
    # Define the compute resources
    resources = {
        "cpu": 4,
        "memory": "16GB",
        "gpu": 1 if model_type == "lstm" else 0
    }
    
    # Define the command to run
    command = f"cd /workspace && pip install -r thundercompute/requirements-thunder.txt && python thundercompute/train_scripts/train_{model_type}.py"
    
    # Create the job
    job_id = create_job(
        name=f"finpulse-{model_type}-training",
        command=command,
        resources=resources,
        environment_variables={
            "DATA_ID": data_id,
            "MODEL_TYPE": model_type
        },
        timeout_hours=2
    )
    
    print(f"Job submitted with ID: {job_id}")
    
    # Poll for job completion
    status = "PENDING"
    while status in ["PENDING", "RUNNING"]:
        time.sleep(60)  # Check every minute
        status = get_job_status(job_id)
        print(f"Current status: {status}")
    
    if status == "COMPLETED":
        # Download the trained model
        model_path = f"finpulse/models/{model_type}_model.joblib"
        local_path = download_model(model_path, f"models/{model_type}_model.joblib")
        print(f"Model downloaded to {local_path}")
        return local_path
    else:
        print(f"Job failed with status: {status}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training job on ThunderCompute')
    parser.add_argument('--model', type=str, required=True, choices=['xgboost', 'lightgbm', 'lstm'],
                      help='Model type to train')
    parser.add_argument('--data-id', type=str, required=True,
                      help='Data ID on ThunderCompute')
    
    args = parser.parse_args()
    run_training_job(args.model, args.data_id)
```

## 6. Using ThunderCompute

### Train a Single Model
```bash
# Train XGBoost model
python thundercompute/run.py --model xgboost --data-id YOUR_DATA_ID

# Train LightGBM model
python thundercompute/run.py --model lightgbm --data-id YOUR_DATA_ID

# Train LSTM model
python thundercompute/run.py --model lstm --data-id YOUR_DATA_ID
```

### Run All Training Jobs Sequentially
```bash
for model in xgboost lightgbm lstm; do
    python thundercompute/run.py --model $model --data-id YOUR_DATA_ID
done
```

## 7. Model Deployment and Integration

### Integrating the Models into Your Application
Once models are trained and downloaded:

1. Load all models into your application:
```python
import joblib

xgboost_model = joblib.load('models/xgboost_model.joblib')
lightgbm_model = joblib.load('models/lightgbm_model.joblib')
lstm_model = joblib.load('models/lstm_model.joblib')
```

2. Create an ensemble prediction function:
```python
def ensemble_predict(data, weights={'xgboost': 0.4, 'lightgbm': 0.4, 'lstm': 0.2}):
    # Prepare data for each model type
    lstm_data = prepare_data_for_lstm(data)
    tree_data = prepare_data_for_trees(data)
    
    # Get predictions
    xgb_pred = xgboost_model.predict(tree_data)
    lgbm_pred = lightgbm_model.predict(tree_data)
    lstm_pred = lstm_model.predict(lstm_data)
    
    # Weighted ensemble
    final_pred = (weights['xgboost'] * xgb_pred + 
                  weights['lightgbm'] * lgbm_pred + 
                  weights['lstm'] * lstm_pred)
                  
    return final_pred
```

## 8. Cost Optimization

ThunderCompute's free tier has limitations. To optimize usage:

1. **Batch Training**: Train models less frequently with larger data batches
2. **Checkpointing**: Save intermediate results to resume training if interrupted
3. **Model Pruning**: Use techniques like quantization to reduce model size
4. **Job Scheduling**: Schedule jobs during off-peak hours for better resource allocation
5. **Incremental Learning**: When possible, update existing models instead of retraining

## 9. Monitoring and Maintenance

1. Set up a monitoring script to track:
   - Training job status
   - Model performance metrics
   - API usage and quotas

2. Create an automated retraining pipeline that:
   - Detects model drift
   - Schedules retraining when performance degrades
   - Keeps a version history of models

## 10. Troubleshooting Common Issues

1. **Connection Issues**:
   - Verify API key is correctly configured
   - Check network connectivity
   - Ensure you're within rate limits

2. **Job Failures**:
   - Check logs for Python exceptions
   - Verify resource allocation is sufficient
   - Check for timeout issues on long-running jobs

3. **Performance Issues**:
   - Try reducing model complexity
   - Use smaller datasets for initial testing
   - Enable checkpointing for long-running jobs