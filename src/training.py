"""
training.py
Uses AutoGluon TabularPredictor for automated model training.
AutoGluon automatically handles:
  - Problem type detection (classification / regression)
  - Model selection with lightweight-first progressive strategy
  - Hyperparameter tuning
  - Ensemble building and stacking
  - Cross-validation
"""

import os
import shutil
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
import warnings
warnings.filterwarnings('ignore')


# Quality presets mapping: maps user-facing names to AutoGluon presets
# Lightweight-first: default uses only fast, low-resource models
QUALITY_PRESETS = {
    'lightweight': 'medium_quality_faster_train',
    'balanced': 'medium_quality',
    'best': 'best_quality',
}

# Restrict model types for lightweight mode so it only uses lightweight models
# before escalating to more sophisticated / resource-intensive models
LIGHTWEIGHT_HYPERPARAMETERS = {
    'GBM': {},            # Lightweight Gradient Boosting
    'CAT': {},            # CatBoost (handles categoricals well, small footprint)
    'RF': {'n_estimators': 100},  # Random Forest (lightweight baseline)
    'LR': {},             # Linear model (fastest, simplest)
    'XGB': {},            # XGBoost (lightweight config)
}

BALANCED_HYPERPARAMETERS = {
    'GBM': {},
    'CAT': {},
    'RF': {},
    'XGB': {},
    'XT': {},             # Extra Trees
    'KNN': {},            # K-Nearest Neighbors
    'LR': {},
}

# For 'best' quality, we don't restrict hyperparameters — let AutoGluon decide


def run_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series,
                 problem_type: str, quality: str = 'lightweight',
                 time_limit: int = None) -> dict:
    """
    Train models using AutoGluon TabularPredictor.

    Parameters
    ----------
    X_train, y_train : Training features and target.
    X_test, y_test : Test features and target.
    problem_type : 'classification' or 'regression'.
    quality : One of 'lightweight', 'balanced', 'best'.
              Lightweight-first: starts with fast, low-resource models.
    time_limit : Max training time in seconds (None = no limit).

    Returns
    -------
    dict containing:
      - predictor: the trained TabularPredictor
      - leaderboard: DataFrame with all model results
      - model_name: name of the best model
      - problem_type: detected problem type
    """
    # Build the training DataFrame
    train_data = X_train.copy()
    train_data['__target__'] = y_train.values

    test_data = X_test.copy()
    test_data['__target__'] = y_test.values

    # Map quality string to AutoGluon preset
    preset = QUALITY_PRESETS.get(quality, 'medium_quality_faster_train')

    # Determine hyperparameters based on quality
    hyperparameters = None
    if quality == 'lightweight':
        hyperparameters = LIGHTWEIGHT_HYPERPARAMETERS
    elif quality == 'balanced':
        hyperparameters = BALANCED_HYPERPARAMETERS
    # 'best' uses AutoGluon's default (all models, full search)

    # Use a temp directory for AutoGluon model output (cleaned up after)
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'autogluon_models')
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # Determine AutoGluon problem type
    # AutoGluon uses: 'binary', 'multiclass', 'regression', 'quantile'
    # Our pipeline uses: 'classification', 'regression'
    if problem_type == 'classification':
        # Determine if binary or multiclass based on target unique count
        unique_count = y_train.nunique()
        ag_problem_type = 'binary' if unique_count == 2 else 'multiclass'
    else:
        ag_problem_type = 'regression'

    # For evaluation metric, align with standard names
    eval_metric = 'accuracy' if problem_type == 'classification' else 'r2'

    try:
        predictor = TabularPredictor(
            label='__target__',
            path=model_dir,
            problem_type=ag_problem_type,
            eval_metric=eval_metric,
        )

        predictor.fit(
            train_data=TabularDataset(train_data),
            presets=preset,
            hyperparameters=hyperparameters,
            time_limit=time_limit,
            verbosity=0,
        )

        # Get leaderboard on test data
        leaderboard = predictor.leaderboard(test_data, silent=True)

        # Get best model name
        model_name = predictor.model_best

    except Exception as e:
        # Clean up on error
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        raise e

    return {
        'predictor': predictor,
        'leaderboard': leaderboard,
        'model_name': model_name,
        'problem_type': problem_type,
        'model_dir': model_dir,
    }


def get_model_summary(predictor) -> dict:
    """
    Return a summary of all trained models from the predictor.
    """
    leaderboard = predictor.leaderboard(silent=True)
    summary = {
        'num_models': len(leaderboard),
        'best_model': predictor.model_best,
        'leaderboard': leaderboard,
    }
    return summary


def cleanup(predictor_or_dir):
    """
    Clean up AutoGluon model directory after pipeline completes.
    """
    if isinstance(predictor_or_dir, str):
        model_dir = predictor_or_dir
    else:
        model_dir = predictor_or_dir.path

    if model_dir and os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)