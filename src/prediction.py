"""
prediction.py
Uses AutoGluon's built-in evaluation for predictions, metrics, and feature importance.
AutoGluon handles all metric calculations natively.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')


def evaluate_predictor(predictor, X_test, y_test, problem_type: str,
                       feature_names: list = None) -> dict:
    """
    Evaluate an AutoGluon TabularPredictor on test data.

    Parameters
    ----------
    predictor : Trained TabularPredictor.
    X_test, y_test : Test features and target.
    problem_type : 'classification' or 'regression'.
    feature_names : List of feature column names.

    Returns
    -------
    dict containing:
      - all_metrics: dict mapping model_name -> metrics dict
      - best_model_name: name of the best model
      - leaderboard: AutoGluon leaderboard DataFrame
      - feature_importance: dict mapping model_name -> feature importance list
    """
    # Build test DataFrame with target
    test_data = X_test.copy()
    test_data['__target__'] = y_test.values

    # Get leaderboard
    leaderboard = predictor.leaderboard(test_data, silent=True)
    leaderboard = leaderboard.reset_index(drop=True)

    # Get predictions for all models
    all_metrics = {}

    # Predict with the best model
    y_pred = predictor.predict(test_data)

    # Get per-model predictions from the leaderboard
    for idx, row in leaderboard.iterrows():
        model_name = row['model']

        try:
            # Get predictions for this specific model
            model_pred = predictor.predict(test_data, model=model_name)

            if problem_type == 'classification':
                metrics = _evaluate_classification(
                    y_test, model_pred, model_name
                )
            else:
                metrics = _evaluate_regression(
                    y_test, model_pred, model_name
                )

            # Add AutoGluon's score from leaderboard
            score_col = [c for c in leaderboard.columns
                         if c not in ['model', 'score_val', 'pred_time_val',
                                      'fit_time', 'pred_time_test', 'fit_time_marginal',
                                      'pred_time_val_marginal', 'pred_time_test_marginal',
                                      'stack_level', 'can_infer', 'fit_order']]
            if score_col:
                score_name = score_col[0]
                metrics['leaderboard_score'] = row.get(score_name, row.get('score_test', None))

            all_metrics[model_name] = metrics

        except Exception:
            # Skip models that fail to predict
            continue

    # Also compute best model metrics explicitly
    best_model_name = predictor.model_best
    if best_model_name and best_model_name not in all_metrics:
        try:
            best_pred = predictor.predict(test_data, model=best_model_name)
            if problem_type == 'classification':
                all_metrics[best_model_name] = _evaluate_classification(
                    y_test, best_pred, best_model_name
                )
            else:
                all_metrics[best_model_name] = _evaluate_regression(
                    y_test, best_pred, best_model_name
                )
        except Exception:
            pass

    # Feature importance (permutation-based, via AutoGluon)
    feature_importance = {}
    if feature_names is not None:
        try:
            # AutoGluon's built-in feature importance
            fi_df = predictor.feature_importance(test_data, silent=True)
            if fi_df is not None and not fi_df.empty:
                importance_list = []
                for feat_idx, feat_row in fi_df.iterrows():
                    importance_list.append((
                        feat_idx,
                        feat_row.get('importance', feat_row.get('mean_importance', 0))
                    ))
                importance_list.sort(key=lambda x: abs(x[1]), reverse=True)
                feature_importance[best_model_name] = importance_list
        except Exception:
            pass

    return {
        'all_metrics': all_metrics,
        'best_model_name': best_model_name,
        'leaderboard': leaderboard,
        'feature_importance': feature_importance,
    }


def _evaluate_classification(y_true, y_pred, model_name: str) -> dict:
    """Compute classification metrics."""
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        'y_pred': y_pred,
        'y_true': y_true,
    }
    return metrics


def _evaluate_regression(y_true, y_pred, model_name: str) -> dict:
    """Compute regression metrics."""
    metrics = {
        'model_name': model_name,
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'y_pred': y_pred,
        'y_true': y_true,
        'residuals': (y_true - y_pred).tolist(),
    }
    return metrics


def find_best_model(all_metrics: dict, problem_type: str,
                    best_model_name_hint: str = None) -> tuple:
    """
    Find the best performing model from the metrics dict.
    Uses AutoGluon's best model name hint if available.

    Returns (model_name, metric_dict).
    """
    if not all_metrics:
        return None, None

    # If AutoGluon's best model is available, use it
    if best_model_name_hint and best_model_name_hint in all_metrics:
        return best_model_name_hint, all_metrics[best_model_name_hint]

    # Fallback: score-based selection (same as original)
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


def run_pipeline(predictor, X_test, y_test, problem_type: str,
                 feature_names: list = None, best_model_hint: str = None) -> dict:
    """
    Run the full evaluation pipeline using AutoGluon predictor.

    This replaces the old run_pipeline that took trained_models dict.
    """
    eval_results = evaluate_predictor(
        predictor, X_test, y_test, problem_type, feature_names
    )

    best_model_name, best_metrics = find_best_model(
        eval_results['all_metrics'],
        problem_type,
        best_model_name_hint=best_model_hint or eval_results.get('best_model_name')
    )

    eval_results['best_model_name'] = best_model_name

    return eval_results
