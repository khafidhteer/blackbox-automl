"""
data_split.py
Handles feature/target separation and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Automatically detect whether this is a classification or regression problem.
    - Classification: target is categorical, object, bool, or numeric with <= 20 unique values
    - Regression: target is numeric with > 20 unique values
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    y = df[target_col]

    if pd.api.types.is_numeric_dtype(y):
        unique_count = y.nunique()
        if unique_count <= 20:
            return 'classification'
        else:
            return 'regression'
    else:
        # Categorical, object, or bool → classification
        return 'classification'


def auto_detect_target(df: pd.DataFrame) -> str:
    """
    Automatically detect the target column.
    Heuristic: last column is usually the target in ML datasets.
    If the last column is numeric with few unique values or categorical, use it.
    """
    # Default to last column
    target_col = df.columns[-1]

    # Special common target column names
    common_targets = ['target', 'label', 'y', 'class', 'output', 'result', 'outcome',
                      'price', 'cost', 'value', 'score', 'category', 'type']
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in common_targets:
            target_col = col
            break

    return target_col


def split_features_target(df: pd.DataFrame, target_col: str = None) -> tuple:
    """
    Split DataFrame into X (features) and y (target).
    Automatically exclude non-numeric columns from features for model training,
    but keep track of them for reference.
    """
    if target_col is None:
        target_col = auto_detect_target(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Track which columns are numeric vs non-numeric
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # For modeling, one-hot encode categorical features
    if categorical_cols:
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        X_encoded = X

    return X, X_encoded, y, target_col, numeric_cols, categorical_cols


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                     random_state: int = 42, stratify=None) -> tuple:
    """
    Split data into training and testing sets.
    Automatically applies stratification for classification problems.
    """
    if stratify is None:
        # Check if classification (few unique values)
        if y.nunique() <= 20 and pd.api.types.is_numeric_dtype(y):
            stratify = y
        elif not pd.api.types.is_numeric_dtype(y):
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=stratify
    )

    split_info = {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape,
        'y_test_shape': y_test.shape,
        'train_percentage': round((1 - test_size) * 100, 1),
        'test_percentage': round(test_size * 100, 1),
        'stratified': stratify is not None
    }

    return X_train, X_test, y_train, y_test, split_info


def run_pipeline(df: pd.DataFrame, target_col: str = None, test_size: float = 0.2,
                 random_state: int = 42) -> dict:
    """
    Run the full data splitting pipeline.
    Returns a dict with all results.
    """
    # Detect problem type
    if target_col is None:
        target_col = auto_detect_target(df)
    problem_type = detect_problem_type(df, target_col)

    # Split features and target
    X, X_encoded, y, detected_target, numeric_cols, categorical_cols = split_features_target(df, target_col)

    # Train/test split
    X_train, X_test, y_train, y_test, split_info = split_train_test(
        X_encoded, y, test_size=test_size, random_state=random_state
    )

    return {
        'X': X,
        'X_encoded': X_encoded,
        'y': y,
        'target_col': detected_target,
        'problem_type': problem_type,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'split_info': split_info
    }