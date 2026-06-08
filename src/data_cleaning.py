"""
data_cleaning.py
Handles data cleansing: missing values, duplicates, outliers, basic statistics.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file and return a DataFrame."""
    df = pd.read_csv(file_path)
    return df


def basic_info(df: pd.DataFrame) -> dict:
    """Return basic info about the dataset as a dict for notebook display."""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_count': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicate_count': df.duplicated().sum(),
        'describe': df.describe(include='all').to_dict()
    }
    return info


def handle_missing_values(df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
    """
    Handle missing values.
    - auto: numerical → median, categorical → mode
    - mean / median / mode: force specific strategy
    - drop: drop rows with any missing values
    """
    df_clean = df.copy()
    steps = []

    if strategy == 'drop':
        before = len(df_clean)
        df_clean = df_clean.dropna()
        after = len(df_clean)
        steps.append(f"Dropped {before - after} rows with missing values.")
        return df_clean, steps

    for col in df_clean.columns:
        if df_clean[col].isnull().sum() == 0:
            continue

        if strategy == 'auto':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                fill_val = df_clean[col].median()
                method = 'median'
            else:
                fill_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                method = 'mode'
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
            fill_val = df_clean[col].mean()
            method = 'mean'
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
            fill_val = df_clean[col].median()
            method = 'median'
        elif strategy == 'mode':
            fill_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
            method = 'mode'
        else:
            fill_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
            method = 'mode'

        missing_count = df_clean[col].isnull().sum()
        df_clean[col] = df_clean[col].fillna(fill_val)
        steps.append(f"Column '{col}': filled {missing_count} missing values using {method} (value: {fill_val:.4f})"
                     if isinstance(fill_val, float) else
                     f"Column '{col}': filled {missing_count} missing values using {method} (value: {fill_val})")

    return df_clean, steps


def remove_duplicates(df: pd.DataFrame) -> tuple:
    """Remove duplicate rows."""
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    after = len(df_clean)
    steps = [f"Removed {before - after} duplicate rows. Rows before: {before}, after: {after}."]
    return df_clean, steps


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> dict:
    """Detect outliers using IQR method for numerical columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_info = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': round(len(outliers) / len(df) * 100, 2),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    return outlier_info


def handle_outliers(df: pd.DataFrame, method: str = 'cap', columns: list = None) -> tuple:
    """
    Handle outliers.
    - cap: cap at IQR bounds
    - remove: remove outlier rows
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    df_clean = df.copy()
    steps = []

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_count = len(df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)])

        if method == 'cap':
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            steps.append(f"Column '{col}': capped {outlier_count} outliers at [{lower_bound:.4f}, {upper_bound:.4f}].")
        elif method == 'remove':
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            steps.append(f"Column '{col}': removed {outlier_count} outlier rows.")

    return df_clean, steps


def run_pipeline(df: pd.DataFrame, config: dict = None) -> tuple:
    """
    Run the full data cleaning pipeline.
    Returns cleaned DataFrame and list of explanation steps.
    """
    if config is None:
        config = {
            'missing_strategy': 'auto',
            'remove_duplicates': True,
            'outlier_method': 'cap'
        }

    all_steps = []
    df_clean = df.copy()

    # Step 1: Basic info
    info = basic_info(df_clean)

    # Step 2: Handle missing values
    df_clean, missing_steps = handle_missing_values(df_clean, config.get('missing_strategy', 'auto'))
    all_steps.extend(missing_steps)

    # Step 3: Remove duplicates
    if config.get('remove_duplicates', True):
        df_clean, dup_steps = remove_duplicates(df_clean)
        all_steps.extend(dup_steps)

    # Step 4: Handle outliers
    if config.get('outlier_method') and config.get('outlier_method') != 'none':
        df_clean, outlier_steps = handle_outliers(df_clean, config.get('outlier_method', 'cap'))
        all_steps.extend(outlier_steps)

    return df_clean, all_steps, info