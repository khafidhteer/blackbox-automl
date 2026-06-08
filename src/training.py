"""
training.py
Handles model training with automatic problem type detection and multiple models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Try importing xgboost, but don't fail if not available
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_models(problem_type: str) -> dict:
    """
    Return a dictionary of models based on problem type.
    """
    models = {}

    if problem_type == 'classification':
        models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
        models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
        models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)
    else:  # regression
        models['Linear Regression'] = LinearRegression()
        models['Ridge Regression'] = Ridge(alpha=1.0, random_state=42)
        models['Lasso Regression'] = Lasso(alpha=1.0, random_state=42)
        models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    return models


def get_param_grids(problem_type: str) -> dict:
    """
    Return hyperparameter grids for tuning.
    Small grids for reasonable execution time.
    """
    grids = {}

    if problem_type == 'classification':
        grids['Logistic Regression'] = {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']
        }
        grids['Random Forest'] = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        if XGBOOST_AVAILABLE:
            grids['XGBoost'] = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
    else:  # regression
        grids['Random Forest'] = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        if XGBOOST_AVAILABLE:
            grids['XGBoost'] = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }

    return grids


def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train, y_train, problem_type: str, tune: bool = False) -> dict:
    """
    Train multiple models and return trained models with their names.
    If tune=True, perform hyperparameter tuning on applicable models.
    """
    models = get_models(problem_type)
    param_grids = get_param_grids(problem_type)
    trained_models = {}
    training_results = []

    for name, model in models.items():
        result = {'model_name': name}

        try:
            if tune and name in param_grids:
                # Hyperparameter tuning
                grid = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy' if problem_type == 'classification' else 'r2', n_jobs=-1)
                grid.fit(X_train, y_train)
                trained_models[name] = grid.best_estimator_
                result['best_params'] = grid.best_params_
                result['tuned'] = True
            else:
                model.fit(X_train, y_train)
                trained_models[name] = model
                result['tuned'] = False

            result['success'] = True
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            continue

        training_results.append(result)

    return trained_models, training_results


def cross_validate(model, X_train, y_train, problem_type: str, cv: int = 5) -> dict:
    """
    Perform cross-validation and return scores.
    """
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'

    try:
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max()
        }
    except Exception as e:
        return {
            'cv_scores': None,
            'error': str(e)
        }


def run_pipeline(X_train, y_train, X_test, y_test, problem_type: str,
                 tune: bool = False, scale: bool = True) -> dict:
    """
    Run the full training pipeline.
    """
    # Scale features if needed (for SVM, Logistic Regression, Linear models)
    scaler = None
    if scale and problem_type in ['classification']:
        # Scale for models that need it
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test

    # Train models
    trained_models, training_results = train_models(
        X_train_scaled if scale else X_train,
        y_train, problem_type, tune
    )

    # Cross-validation for each model
    cv_results = {}
    for name, model in trained_models.items():
        cv_results[name] = cross_validate(model, X_train_scaled, y_train, problem_type)

    return {
        'trained_models': trained_models,
        'training_results': training_results,
        'cv_results': cv_results,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }