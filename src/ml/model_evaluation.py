"""
Above-industry-grade model evaluation for STOCKER.
Supports classification, regression, drift detection, and full artifact traceability.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, classification_report
)
from src.entity.artifact_entity import ValidationArtifact
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

def evaluate_model(
    model: Any,
    features: pd.DataFrame,
    config: Dict[str, Any],
    baseline_metrics: Optional[Dict[str, float]] = None,
    output_dir: str = "evaluation_artifacts"
) -> Dict[str, Any]:
    """
    Evaluate model performance, detect drift, and save all artifacts.
    Returns a robust evaluation artifact dict.
    """
    logger = logging.getLogger("evaluation")
    os.makedirs(output_dir, exist_ok=True)
    target_col = config.get('target_col', 'target')
    y_true = features[target_col]
    X = features.drop(columns=[target_col])
    y_pred = model.predict(X)
    artifact = {
        "metrics": {},
        "drift": {},
        "plots": {},
        "timestamp": datetime.now().isoformat()
    }
    # Classification or regression?
    task_type = config.get('task_type', 'classification')
    if task_type == 'classification':
        artifact["metrics"] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        # ROC AUC if binary
        if len(np.unique(y_true)) == 2:
            try:
                y_prob = model.predict_proba(X)[:, 1]
                artifact["metrics"]["roc_auc"] = roc_auc_score(y_true, y_prob)
            except Exception:
                pass
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        artifact["plots"]["confusion_matrix"] = cm_path
        # Classification report
        artifact["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
    else:  # regression
        artifact["metrics"] = {
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
    # Feature importance (if available)
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(8, 5))
            plt.title("Feature Importances")
            plt.bar(range(X.shape[1]), importances[indices])
            plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
            plt.tight_layout()
            fi_path = os.path.join(output_dir, "feature_importance.png")
            plt.savefig(fi_path)
            plt.close()
            artifact["plots"]["feature_importance"] = fi_path
    except Exception as e:
        logger.warning(f"Feature importance plotting failed: {e}")
    # Drift detection (performance drop)
    if baseline_metrics is not None:
        drift = {}
        for k, v in artifact["metrics"].items():
            if k in baseline_metrics:
                drift[k] = v - baseline_metrics[k]
        artifact["drift"] = drift
    # Save metrics as JSON
    import json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(artifact["metrics"], f, indent=2)
    artifact["metrics_path"] = metrics_path
    logger.info(f"Evaluation complete. Metrics: {artifact['metrics']}")
    return artifact

"""
Model evaluation component for STOCKER Pro.

This module provides standardized evaluation metrics and comparison tools
for financial prediction models, supporting both regression and classification tasks.
It integrates with the existing evaluation metrics in src/ml/evaluation.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple, Any
import logging
import os
import json
from datetime import datetime

from src.ml.evaluation import (
    calculate_basic_metrics,
    directional_accuracy,
    weighted_directional_accuracy,
    sharpe_ratio,
    sortino_ratio,
    maximum_drawdown,
    calmar_ratio,
    trading_strategy_returns,
    evaluate_financial_model
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Standardized model evaluation for STOCKER Pro.
    
    This class provides methods to evaluate and compare financial prediction models
    using both standard ML metrics and finance-specific metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model evaluator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results = {}
        self.models_evaluated = []
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: Optional[np.ndarray] = None,
        include_strategy_simulation: bool = True,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.0,
        annualization: int = 252
    ) -> Dict[str, float]:
        """
        Evaluate a single model using financial metrics.
        
        Args:
            model_name: Name of the model being evaluated
            y_true: True values (prices or returns)
            y_pred: Predicted values
            prices: Price series (if y_true is returns)
            include_strategy_simulation: Whether to include trading strategy simulation
            transaction_cost: Transaction cost for strategy simulation
            risk_free_rate: Risk-free rate for Sharpe/Sortino ratios
            annualization: Annualization factor (252 for daily, 12 for monthly)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Use the comprehensive evaluation function from ml/evaluation.py
        metrics = evaluate_financial_model(
            y_true=y_true,
            y_pred=y_pred,
            prices=prices,
            include_strategy_simulation=include_strategy_simulation,
            transaction_cost=transaction_cost,
            risk_free_rate=risk_free_rate,
            annualization=annualization
        )
        
        # Store results for this model
        self.results[model_name] = metrics
        if model_name not in self.models_evaluated:
            self.models_evaluated.append(model_name)
        
        return metrics
    
    def compare_models(
        self,
        models_dict: Dict[str, Any] = None,
        test_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_true: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test data.
        
        Args:
            models_dict: Dictionary of model name -> model object
                        If None, uses previously evaluated models
            test_data: Test features (if models_dict is provided)
            y_true: True target values (if models_dict is provided)
            metrics: List of metrics to compare (if None, uses all available)
            
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        if models_dict is not None and test_data is not None and y_true is not None:
            # Evaluate each model on the provided test data
            for name, model in models_dict.items():
                try:
                    y_pred = model.predict(test_data)
                    self.evaluate_model(name, y_true, y_pred)
                except Exception as e:
                    logger.error(f"Error evaluating model {name}: {e}")
        
        # If no models have been evaluated, return empty DataFrame
        if not self.results:
            logger.warning("No models have been evaluated")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison = {}
        for model_name, model_metrics in self.results.items():
            if metrics:
                # Filter to requested metrics
                comparison[model_name] = {k: v for k, v in model_metrics.items() if k in metrics}
            else:
                comparison[model_name] = model_metrics
        
        return pd.DataFrame(comparison).T
    
    def plot_comparison(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comparison of models across selected metrics.
        
        Args:
            metrics: List of metrics to compare (if None, uses key metrics)
            figsize: Figure size
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get comparison data
        if not metrics:
            # Default to key metrics if none specified
            metrics = [
                'rmse', 'mae', 'directional_accuracy', 
                'sharpe_ratio', 'sortino_ratio', 'max_drawdown'
            ]
        
        comparison_df = self.compare_models(metrics=metrics)
        
        if comparison_df.empty:
            logger.warning("No data to plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No models evaluated", ha='center', va='center')
            return fig
        
        # Create plot
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Comparison by {metric}')
                axes[i].set_ylabel(metric)
                
                # Add value labels on bars
                for j, v in enumerate(comparison_df[metric]):
                    if not pd.isna(v):
                        axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    def save_results(self, output_dir: str, filename: Optional[str] = None) -> str:
        """
        Save evaluation results to disk.
        
        Args:
            output_dir: Directory to save results
            filename: Optional filename (default: model_comparison_{timestamp}.json)
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.json"
        
        output_path = os.path.join(output_dir, filename)
        
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {}
        for model, metrics in self.results.items():
            serializable_results[model] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in metrics.items()
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_path}")
        return output_path

def compare_models(
    models: Dict[str, Any],
    test_data: Union[pd.DataFrame, np.ndarray],
    y_true: np.ndarray,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Utility function to compare multiple models on the same test data.
    
    Args:
        models: Dictionary of model name -> model object
        test_data: Test features
        y_true: True target values
        metrics: List of metrics to compare (if None, uses all available)
        
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    evaluator = ModelEvaluator()
    
    for name, model in models.items():
        try:
            y_pred = model.predict(test_data)
            evaluator.evaluate_model(name, y_true, y_pred)
        except Exception as e:
            logger.error(f"Error evaluating model {name}: {e}")
    
    return evaluator.compare_models(metrics=metrics)
