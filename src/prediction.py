"""
prediction.py
Handles model evaluation: predictions, metrics, feature importance, visualizations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')


def evaluate_classification(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a classification model and return metrics.
    """
    y_pred = model.predict(X_test)

    # Handle probability prediction
    try:
        y_prob = model.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = None
    except (AttributeError, ValueError):
        y_prob_pos = None

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        'y_pred': y_pred,
        'y_true': y_test
    }

    # ROC-AUC for binary classification
    if y_prob_pos is not None and len(np.unique(y_test)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob_pos)
            fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        except Exception:
            pass

    return metrics


def evaluate_regression(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a regression model and return metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'model_name': model_name,
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_true': y_test
    }

    # Calculate residuals
    metrics['residuals'] = (y_test - y_pred).tolist()

    return metrics


def get_feature_importance(model, feature_names: list, model_name: str = '') -> list:
    """
    Extract feature importance from tree-based models or coefficients from linear models.
    """
    importance = None

    # Tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    # Linear models (coefficients)
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim == 1:
            importance = model.coef_
        elif model.coef_.ndim == 2:
            # Multi-class, take the mean absolute coefficient
            importance = np.mean(np.abs(model.coef_), axis=0)

    if importance is not None:
        # Create list of (feature, importance) tuples sorted by importance
        feature_importance = list(zip(feature_names, importance))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_importance
    else:
        return []


def run_pipeline(trained_models: dict, X_test, y_test, problem_type: str,
                 feature_names: list = None) -> dict:
    """
    Run the full evaluation pipeline for all models.
    """
    all_metrics = {}
    all_importance = {}

    for name, model in trained_models.items():
        if problem_type == 'classification':
            metrics = evaluate_classification(model, X_test, y_test, name)
        else:
            metrics = evaluate_regression(model, X_test, y_test, name)

        all_metrics[name] = metrics

        # Feature importance
        if feature_names is not None:
            importance = get_feature_importance(model, feature_names, name)
            if importance:
                all_importance[name] = importance

    return {
        'all_metrics': all_metrics,
        'feature_importance': all_importance
    }


def find_best_model(all_metrics: dict, problem_type: str) -> tuple:
    """
    Find the best performing model.
    Returns (model_name, metric_dict).
    """
    if not all_metrics:
        return None, None

    best_model_name = None
    best_score = -np.inf

    for name, metrics in all_metrics.items():
        if problem_type == 'classification':
            score = metrics.get('accuracy', 0)
        else:
            score = metrics.get('r2', -np.inf)

        if score > best_score:
            best_score = score
            best_model_name = name

    return best_model_name, all_metrics.get(best_model_name)