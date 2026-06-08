"""
notebook_generator.py
Generates the final Jupyter Notebook (.ipynb) with all steps explained
**and fully executable Python code cells** — not just a static report.
"""

import json
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_to_base64(fig):
    """Convert a matplotlib figure to base64-encoded string for embedding in notebook."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def create_markdown_cell(source: str) -> dict:
    """Create a markdown cell for the notebook."""
    if isinstance(source, str):
        source = source.split('\n')
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source
    }


def create_code_cell(source: str, outputs: list = None) -> dict:
    """Create a code cell for the notebook."""
    if isinstance(source, str):
        source = source.split('\n')
    cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': outputs or [],
        'source': source
    }
    return cell


def create_text_output(text: str) -> dict:
    """Create a text output for a code cell."""
    return {
        'output_type': 'stream',
        'name': 'stdout',
        'text': [text]
    }


def create_image_output(base64_img: str) -> dict:
    """Create an image output for a code cell."""
    return {
        'output_type': 'display_data',
        'data': {
            'image/png': base64_img,
            'text/plain': ['<matplotlib figure>']
        },
        'metadata': {}
    }


def create_dataframe_output(df: pd.DataFrame, max_rows: int = 10) -> dict:
    """Create a dataframe HTML preview output."""
    html = df.head(max_rows).to_html(classes='table table-striped', border=0, index=True)
    return {
        'output_type': 'execute_result',
        'data': {
            'text/html': [html],
            'text/plain': [df.head(max_rows).to_string()]
        },
        'metadata': {},
        'execution_count': None
    }


def make_config_header(csv_path: str, target_col, problem_type, quality_preset: str,
                       time_limit, seed: int, test_size: float,
                       saved_results_path: str = None) -> list:
    """Create the configuration code cell at the top of the notebook."""
    code = f"""# ========== CONFIGURATION ==========
# Modify these variables to control the pipeline behavior

CSV_PATH = "{csv_path}"              # Path to your CSV file
TARGET_COL = {target_col!r}          # Target column name (None = auto-detect)
PROBLEM_TYPE = {problem_type!r}       # "classification", "regression", or None for auto-detect
QUALITY_PRESET = "{quality_preset}"  # "medium_quality_faster_train", "medium_quality", "best_quality"
TIME_LIMIT = {time_limit!r}           # Max training time in seconds (None = no limit)
RUN_TRAINING = True                  # Set to False to skip re-training and load saved results
SEED = {seed}
TEST_SIZE = {test_size}
# ===================================
"""
    # Build the pre-computed output showing what was used during generation
    output_lines = [
        f"CSV_PATH = '{csv_path}'",
        f"TARGET_COL = {target_col!r}",
        f"PROBLEM_TYPE = {problem_type!r}",
        f"QUALITY_PRESET = '{quality_preset}'",
        f"TIME_LIMIT = {time_limit!r}",
        "RUN_TRAINING = True",
        f"SEED = {seed}",
        f"TEST_SIZE = {test_size}",
    ]
    outputs = [create_text_output("\n".join(output_lines))]

    return [create_code_cell(code, outputs)]


def generate_cleaning_section(df_clean, cleaning_steps, info, csv_path_for_nb: str) -> list:
    """Generate cells for the data cleaning section — fully executable."""
    cells = []

    # Title
    cells.append(create_markdown_cell(
        "## 📋 Step 1: Data Loading & Cleansing\n\n"
        "In this section, we load the dataset and perform initial data cleaning. "
        "Cleaning is a critical step in any machine learning pipeline — "
        "it ensures the data is consistent, complete, and ready for modeling."
    ))

    # --- EXECUTABLE: Load data ---
    code_load = f"""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(CSV_PATH)
print(f"Dataset shape: {{df.shape}}")
print(f"Number of columns: {{df.shape[1]}}")
print(f"Number of rows: {{df.shape[0]}}")
print(f"Number of duplicates: {{df.duplicated().sum()}}")
print(f"\nColumns and their data types:")
print(df.dtypes.to_string())"""
    output_load = create_text_output(
        f"Dataset shape: {info['shape']}\n"
        f"Number of columns: {info['shape'][1]}\n"
        f"Number of rows: {info['shape'][0]}\n"
        f"Number of duplicates: {info['duplicate_count']}\n\n"
        f"Columns and their data types:\n"
        + "\n".join([f"  - {col}: {dtype}" for col, dtype in info['dtypes'].items()])
    )
    cells.append(create_code_cell(code_load, [output_load]))

    # --- EXECUTABLE: Perform cleaning ---
    cells.append(create_markdown_cell(
        "### 🧹 Performing Data Cleaning\n\n"
        "We handle missing values, duplicates, and outliers using standard techniques."
    ))

    code_clean = """# ===== Handle Missing Values =====
# Numerical columns: fill with median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  Column '{col}': filled missing values with median ({df[col].median():.4f})")

# Categorical columns: fill with mode
for col in df.select_dtypes(include=['object', 'category']).columns:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
        df[col].fillna(mode_val, inplace=True)
        print(f"  Column '{col}': filled missing values with mode ('{mode_val}')")

# ===== Remove Duplicates =====
before = len(df)
df.drop_duplicates(inplace=True)
removed = before - len(df)
print(f"\\nRemoved {removed} duplicate rows.")

# ===== Cap Outliers using IQR =====
for col in df.select_dtypes(include=[np.number]).columns:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers_before = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    if outliers_before > 0:
        print(f"  Column '{col}': capped {outliers_before} outliers at [{lower:.4f}, {upper:.4f}]")

print(f"\\n✅ Cleaning complete: {{df.shape[0]}} rows, {{df.shape[1]}} columns")"""

    # Build pre-computed output from actual cleaning steps
    clean_lines = []
    for step in cleaning_steps:
        clean_lines.append(f"  {step}")
    output_clean = create_text_output(
        f"  Dataset shape: {info['shape']}\n"
        + "\n".join(clean_lines) +
        f"\n\n✅ Cleaning complete: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns"
    )
    cells.append(create_code_cell(code_clean, [output_clean]))

    # --- Display first rows ---
    cells.append(create_markdown_cell(
        "### 📄 First 5 Rows of the Dataset\n\n"
        "Let's take a peek at the first few rows to understand the data structure."
    ))

    df_html = df_clean.head().to_html(classes='table', border=0)
    cells.append(create_code_cell(
        "# Display first 5 rows\ndf.head()",
        [{
            'output_type': 'execute_result',
            'data': {
                'text/html': [df_html],
                'text/plain': [df_clean.head().to_string()]
            },
            'metadata': {},
            'execution_count': None
        }]
    ))

    return cells


def generate_eda_section(df_clean, info) -> list:
    """Generate cells for exploratory data analysis — fully executable."""
    cells = []

    cells.append(create_markdown_cell(
        "## 📊 Step 2: Exploratory Data Analysis (EDA)\n\n"
        "EDA helps us understand patterns, relationships, and distributions in the data. "
        "We'll visualize key aspects to gain insights."
    ))

    # --- EXECUTABLE: Statistical summary ---
    cells.append(create_markdown_cell(
        "### 📈 Statistical Summary\n\n"
        "Basic statistical measures for numerical columns (count, mean, std, min, max, quartiles)."
    ))

    desc_df = df_clean.describe(include='all').round(2)
    desc_html = desc_df.to_html(classes='table', border=0)
    cells.append(create_code_cell(
        "# Statistical summary\ndf.describe(include='all')",
        [{
            'output_type': 'execute_result',
            'data': {
                'text/html': [desc_html],
                'text/plain': [desc_df.to_string()]
            },
            'metadata': {},
            'execution_count': None
        }]
    ))

    # --- EXECUTABLE: Correlation heatmap ---
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        cells.append(create_markdown_cell(
            "### 🔥 Correlation Heatmap\n\n"
            "Correlation shows how features relate to each other. "
            "Values near +1 or -1 indicate strong relationships, while values near 0 indicate weak relationships."
        ))

        fig, ax = plt.subplots(figsize=(12, 8))
        corr = df_clean[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        img_b64 = plot_to_base64(fig)

        cells.append(create_code_cell(
            """# Correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax)
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.show()""",
            [create_image_output(img_b64)]
        ))

    # --- EXECUTABLE: Distribution plots ---
    numeric_cols_for_plots = df_clean.select_dtypes(include=[np.number]).columns[:6]
    if len(numeric_cols_for_plots) > 0:
        cells.append(create_markdown_cell(
            "### 📊 Feature Distributions\n\n"
            "Histograms show the distribution of each numerical feature. "
            "This helps identify skewness, multimodal distributions, and potential data quality issues."
        ))

        n_cols = min(3, len(numeric_cols_for_plots))
        n_rows = (len(numeric_cols_for_plots) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(numeric_cols_for_plots):
            if i < len(axes):
                sns.histplot(df_clean[col].dropna(), kde=True, bins=30, ax=axes[i], color='steelblue')
                axes[i].set_title(f'Distribution of {col}', fontsize=12)
                axes[i].set_xlabel(col)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        img_b64 = plot_to_base64(fig)

        cells.append(create_code_cell(
            """# Distribution histograms for numerical features
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
if len(numeric_cols) > 0:
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i], color='steelblue')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()""",
            [create_image_output(img_b64)]
        ))

    return cells


def generate_split_section(split_data: dict) -> list:
    """Generate cells for the data splitting section — fully executable."""
    cells = []

    split_info = split_data['split_info']
    problem_type = split_data['problem_type']

    cells.append(create_markdown_cell(
        "## 🔀 Step 3: Data Splitting\n\n"
        "We split the data into:\n"
        "- **Features (X)**: The input variables used for prediction\n"
        "- **Target (y)**: The value we want to predict\n\n"
        "Then we split into training and testing sets to evaluate model performance on unseen data."
    ))

    # --- EXECUTABLE: Auto-detect target & split features/target ---
    target_col = split_data['target_col']
    codes_split = f"""from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

# Auto-detect target column
if TARGET_COL:
    target_col = TARGET_COL
else:
    possible_targets = ['target', 'label', 'class', 'y', 'outcome', 'result', df.columns[-1]]
    target_col = next((c for c in possible_targets if c in df.columns), df.columns[-1])

print(f"🎯 Target column: {{target_col}}")

y = df[target_col]
X = df.drop(columns=[target_col])

# Drop high-cardinality categorical columns (likely IDs, emails, etc.)
high_card = [c for c in X.select_dtypes(include=['object']).columns if X[c].nunique() > 100]
if high_card:
    X.drop(columns=high_card, inplace=True, errors='ignore')
    print(f"  Dropped high-cardinality columns: {{high_card}}")

print(f"Features shape: {{X.shape}}")
print(f"Target shape: {{y.shape}}")

# Auto-detect problem type
if PROBLEM_TYPE:
    problem_type = PROBLEM_TYPE
else:
    unique_vals = y.nunique()
    if is_numeric_dtype(y) and unique_vals > 10:
        problem_type = 'regression'
    elif unique_vals == 2:
        problem_type = 'binary'
    else:
        problem_type = 'multiclass'
print(f"Problem type: {{problem_type}}")"""
    output_split = create_text_output(
        f"🎯 Target column: {target_col}\n"
        f"  (No high-cardinality columns dropped)\n"
        f"Features shape: {split_data['X_encoded'].shape}\n"
        f"Target shape: {split_data['y'].shape}\n"
        f"Problem type: {problem_type}"
    )
    cells.append(create_code_cell(codes_split, [output_split]))

    # --- Target distribution ---
    y = split_data['y']
    cells.append(create_markdown_cell(
        "### 📊 Target Variable Distribution\n\n"
        "Understanding the distribution of the target variable helps us choose the right evaluation metrics."
    ))

    if problem_type == 'classification':
        fig, ax = plt.subplots(figsize=(8, 5))
        value_counts = y.value_counts()
        colors = sns.color_palette('viridis', n_colors=len(value_counts))
        ax.bar(value_counts.index.astype(str), value_counts.values, color=colors)
        ax.set_title('Target Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        img_b64 = plot_to_base64(fig)

        cells.append(create_code_cell(
            """# Target distribution (classification)
import matplotlib.pyplot as plt
import seaborn as sns

value_counts = y.value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette('viridis', n_colors=len(value_counts))
ax.bar(value_counts.index.astype(str), value_counts.values, color=colors)
ax.set_title('Target Class Distribution', fontsize=14)
ax.set_xlabel('Class')
ax.set_ylabel('Count')
for i, v in enumerate(value_counts.values):
    ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()""",
            [create_image_output(img_b64)]
        ))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(y, kde=True, bins=30, color='steelblue', ax=ax)
        ax.set_title('Target Variable Distribution (Regression)', fontsize=14, fontweight='bold')
        ax.set_xlabel(split_data['target_col'])
        plt.tight_layout()
        img_b64 = plot_to_base64(fig)

        cells.append(create_code_cell(
            f"""# Target distribution (regression)
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(y, kde=True, bins=30, color='steelblue', ax=ax)
ax.set_title('Target Variable Distribution')
ax.set_xlabel('{split_data['target_col']}')
plt.show()""",
            [create_image_output(img_b64)]
        ))

    # --- EXECUTABLE: Train/test split ---
    stratified_text = " (stratified)" if split_info['stratified'] else ""
    cells.append(create_markdown_cell(
        "### ✂️ Train/Test Split\n\n"
        f"We split the data into training and testing sets.{stratified_text}"
    ))

    codes_tts = """# Train/test split
stratify = y if problem_type in ('binary', 'multiclass') else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=stratify
)
print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Testing set:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
print(f"{'✅ Stratified split (class proportions preserved)' if stratify is not None else ''}")"""
    output_tts = create_text_output(
        f"Training set: {split_info['X_train_shape'][0]} samples, {split_info['X_train_shape'][1]} features\n"
        f"Testing set:  {split_info['X_test_shape'][0]} samples, {split_info['X_test_shape'][1]} features\n"
        f"{'✅ Stratified split (class proportions preserved)' if split_info['stratified'] else ''}"
    )
    cells.append(create_code_cell(codes_tts, [output_tts]))

    return cells


def generate_training_section(training_results: dict, cv_results: dict, split_data: dict,
                               quality_preset: str, time_limit) -> list:
    """Generate cells for the model training section — fully executable."""
    cells = []
    problem_type = split_data['problem_type']
    is_autogluon = 'predictor' in training_results or 'leaderboard' in training_results

    cells.append(create_markdown_cell(
        f"## 🧠 Step 4: Model Training\n\n"
        f"Training multiple {problem_type} models to find the best performer. "
        "We use **AutoGluon** — an automated ML framework that handles model selection, "
        "hyperparameter tuning, ensembling, and cross-validation automatically."
    ))

    lb = training_results.get('leaderboard')
    if is_autogluon and lb is not None:
        num_models = len(lb)
        best_model = training_results.get('model_name', 'N/A')
        model_list = "\n".join([f"- {row['model']}" for _, row in lb.iterrows()])
        cells.append(create_markdown_cell(
            "### 🤖 Models Trained (AutoGluon)\n\n"
            f"A total of **{num_models} models** were trained:\n\n"
            f"{model_list}\n\n"
            f"🏆 **Best Model**: `{best_model}`"
        ))

    # --- EXECUTABLE: AutoGluon training ---
    if is_autogluon and lb is not None:
        time_limit_str = str(time_limit) if time_limit is not None else "None"
        code_train = f"""from autogluon.tabular import TabularPredictor, TabularDataset
import os

# Train AutoGluon models
if RUN_TRAINING:
    train_data = X_train.copy()
    train_data[target_col] = y_train

    predictor = TabularPredictor(
        label=target_col,
        problem_type=problem_type,  # binary / multiclass / regression
        eval_metric='accuracy' if problem_type in ('binary', 'multiclass') else 'r2',
    )
    predictor.fit(
        train_data=TabularDataset(train_data),
        presets=QUALITY_PRESET,
        time_limit=TIME_LIMIT if TIME_LIMIT else None,
        verbosity=0,
    )

    # Evaluate on test set
    test_data = X_test.copy()
    test_data[target_col] = y_test
    leaderboard = predictor.leaderboard(test_data, silent=True)
    print(f"✅ Trained {{len(leaderboard)}} models")
    print(f"🏆 Best model: {{predictor.model_best}}")
else:
    # Load pre-computed results from the pipeline run
    import pickle
    saved_path = os.path.join(os.path.dirname(CSV_PATH), '..', 'output', 'pipeline_state.pkl')
    with open(saved_path, 'rb') as f:
        saved = pickle.load(f)
    predictor = saved['predictor']
    leaderboard = saved['leaderboard']
    print(f"✅ Loaded {{len(leaderboard)}} pre-computed models")
    print(f"🏆 Best model: {{predictor.model_best}}")"""
        output_train = create_text_output(
            f"✅ Trained {num_models} models\n"
            f"🏆 Best model: {best_model}"
        )
        cells.append(create_code_cell(code_train, [output_train]))

        # --- EXECUTABLE: Leaderboard display ---
        cells.append(create_markdown_cell(
            "### 📊 AutoGluon Leaderboard\n\n"
            "The leaderboard shows all trained models ranked by performance. "
            "AutoGluon automatically applies cross-validation and ranks models "
            "by their validation score."
        ))

        display_cols = [c for c in lb.columns
                        if c not in ['fit_time_marginal', 'pred_time_val_marginal',
                                     'pred_time_test_marginal', 'stack_level',
                                     'can_infer', 'fit_order']]
        lb_display = lb[display_cols].round(4) if len(display_cols) > 0 else lb.round(4)
        lb_html = lb_display.to_html(classes='table table-striped', border=0, index=False)

        cells.append(create_code_cell(
            """# AutoGluon leaderboard — all model results sorted by performance
leaderboard""",
            [{
                'output_type': 'execute_result',
                'data': {
                    'text/html': [lb_html],
                    'text/plain': [lb_display.to_string()]
                },
                'metadata': {},
                'execution_count': None
            }]
        ))

        # --- EXECUTABLE: Leaderboard bar chart ---
        score_cols = [c for c in lb.columns
                      if c not in ['model', 'score_val', 'pred_time_val',
                                   'fit_time', 'pred_time_test',
                                   'fit_time_marginal', 'pred_time_val_marginal',
                                   'pred_time_test_marginal', 'stack_level',
                                   'can_infer', 'fit_order']]
        score_name = score_cols[0] if score_cols else 'score_test'

        top_n = min(10, len(lb))
        top_models = lb.head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
        model_names = list(top_models['model'])
        scores = [float(top_models[score_name].iloc[i]) for i in range(len(top_models))]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
        bars = ax.barh(range(len(model_names)), scores, color=colors)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Test Score')
        ax.set_title('AutoGluon Model Leaderboard (Top {})'.format(top_n),
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{score:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        img_b64 = plot_to_base64(fig)

        cells.append(create_code_cell(
            """# Leaderboard visualization — bar chart of top models
import matplotlib.pyplot as plt
import numpy as np

top_n = min(10, len(leaderboard))
top_models = leaderboard.head(top_n)
models = top_models['model'].tolist()

# Find the score column (first non-metadata column)
score_cols = [c for c in leaderboard.columns if c not in [
    'model', 'score_val', 'pred_time_val', 'fit_time', 'pred_time_test',
    'fit_time_marginal', 'pred_time_val_marginal', 'pred_time_test_marginal',
    'stack_level', 'can_infer', 'fit_order']]
score_name = score_cols[0] if score_cols else 'score_test'
scores = top_models[score_name].tolist()
scores = [float(s) if s is not None else 0.0 for s in scores]

fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
bars = ax.barh(range(len(models)), scores, color=colors)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models)
ax.set_xlabel('Test Score')
ax.set_title(f'AutoGluon Model Leaderboard (Top {top_n})', fontsize=14)
ax.invert_yaxis()
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f'{score:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.show()""",
            [create_image_output(img_b64)]
        ))

    else:
        # Legacy path — keep minimal
        if isinstance(training_results, list):
            cells.append(create_markdown_cell(
                "### 🤖 Models Trained\n\n"
                + "\n".join([f"- {r['model_name']}" for r in training_results if r.get('success')])
            ))

    return cells


def generate_evaluation_section(eval_results: dict, split_data: dict) -> list:
    """Generate cells for the evaluation section — fully executable."""
    cells = []
    all_metrics = eval_results['all_metrics']
    problem_type = split_data['problem_type']

    cells.append(create_markdown_cell(
        "## 📈 Step 5: Model Evaluation\n\n"
        "Evaluating each model on the held-out test set to compare real-world performance."
    ))

    if problem_type in ('classification', 'binary', 'multiclass'):
        # --- EXECUTABLE: Classification metrics ---
        metrics_data = []
        for name, metrics in all_metrics.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                'Precision': f"{metrics.get('precision', 0):.4f}",
                'Recall': f"{metrics.get('recall', 0):.4f}",
                'F1 Score': f"{metrics.get('f1_score', 0):.4f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 'N/A')}" if metrics.get('roc_auc') is not None else 'N/A'
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_html = metrics_df.to_html(classes='table table-striped', border=0, index=False)

        cells.append(create_code_cell(
            """# Classification metrics for all models
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

if problem_type in ('binary', 'multiclass'):
    metrics_list = []
    for model_name in leaderboard['model'].tolist():
        try:
            y_pred = predictor.predict(X_test, model=model_name)
            row = {
                'Model': model_name,
                'Accuracy': f"{accuracy_score(y_test, y_pred):.4f}",
                'Precision': f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
                'Recall': f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
                'F1 Score': f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}",
            }
            if problem_type == 'binary' and len(np.unique(y_test)) == 2:
                try:
                    y_prob = predictor.predict_proba(X_test, model=model_name)
                    row['ROC-AUC'] = f"{roc_auc_score(y_test, y_prob.iloc[:, 1] if hasattr(y_prob, 'iloc') else y_prob[:, 1]):.4f}"
                except Exception:
                    row['ROC-AUC'] = 'N/A'
            metrics_list.append(row)
        except Exception:
            pass
    pd.DataFrame(metrics_list)""",
            [{
                'output_type': 'execute_result',
                'data': {
                    'text/html': [metrics_html],
                    'text/plain': [metrics_df.to_string()]
                },
                'metadata': {},
                'execution_count': None
            }]
        ))

        # --- EXECUTABLE: Confusion matrix ---
        best_model_name, best_metrics = find_best_model_local(all_metrics, problem_type)
        if best_model_name and 'confusion_matrix' in best_metrics:
            cells.append(create_markdown_cell(
                f"### 🔍 Confusion Matrix — {best_model_name}\n\n"
                "The confusion matrix shows correct vs incorrect predictions for each class."
            ))

            cm = np.array(best_metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            plt.tight_layout()
            img_b64 = plot_to_base64(fig)

            cells.append(create_code_cell(
                """# Confusion matrix for the best model
from sklearn.metrics import confusion_matrix
import seaborn as sns

best_model = predictor.model_best
y_pred = predictor.predict(X_test, model=best_model)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Confusion Matrix — {best_model}', fontsize=14)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.show()""",
                [create_image_output(img_b64)]
            ))

    else:
        # --- EXECUTABLE: Regression metrics ---
        metrics_data = []
        for name, metrics in all_metrics.items():
            metrics_data.append({
                'Model': name,
                'R²': f"{metrics.get('r2', 0):.4f}",
                'MAE': f"{metrics.get('mae', 0):.4f}",
                'RMSE': f"{metrics.get('rmse', 0):.4f}",
                'MSE': f"{metrics.get('mse', 0):.4f}"
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_html = metrics_df.to_html(classes='table table-striped', border=0, index=False)

        cells.append(create_code_cell(
            """# Regression metrics for all models
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

if problem_type == 'regression':
    metrics_list = []
    for model_name in leaderboard['model'].tolist():
        try:
            y_pred = predictor.predict(X_test, model=model_name)
            metrics_list.append({
                'Model': model_name,
                'R²': f"{r2_score(y_test, y_pred):.4f}",
                'MAE': f"{mean_absolute_error(y_test, y_pred):.4f}",
                'RMSE': f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}",
                'MSE': f"{mean_squared_error(y_test, y_pred):.4f}",
            })
        except Exception:
            pass
    pd.DataFrame(metrics_list)""",
            [{
                'output_type': 'execute_result',
                'data': {
                    'text/html': [metrics_html],
                    'text/plain': [metrics_df.to_string()]
                },
                'metadata': {},
                'execution_count': None
            }]
        ))

        # --- EXECUTABLE: Residual plot ---
        best_model_name, best_metrics = find_best_model_local(all_metrics, problem_type)
        if best_model_name and 'residuals' in best_metrics:
            cells.append(create_markdown_cell(
                f"### 📉 Residual Analysis — {best_model_name}\n\n"
                "Residuals (actual - predicted) should be randomly distributed around zero. "
                "Patterns in residuals may indicate model bias or missing features."
            ))

            residuals = np.array(best_metrics['residuals'])
            y_pred = np.array(best_metrics['y_pred'])
            y_true = np.array(best_metrics['y_true'])

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].scatter(y_pred, residuals, alpha=0.6, color='steelblue')
            axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')

            axes[1].hist(residuals, bins=30, edgecolor='white', color='steelblue', alpha=0.7)
            axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')

            plt.tight_layout()
            img_b64 = plot_to_base64(fig)

            cells.append(create_code_cell(
                """# Residual plot for the best regression model
import numpy as np

best_model = predictor.model_best
y_pred = predictor.predict(X_test, model=best_model)
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.6, color='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Predicted', fontsize=12)

axes[1].hist(residuals, bins=30, edgecolor='white', color='steelblue', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals', fontsize=12)
plt.tight_layout()
plt.show()""",
                [create_image_output(img_b64)]
            ))

    # --- EXECUTABLE: Feature importance ---
    if 'feature_importance' in eval_results and eval_results['feature_importance']:
        for model_name, features in eval_results['feature_importance'].items():
            if features:
                cells.append(create_markdown_cell(
                    f"### ⭐ Feature Importance — {model_name}\n\n"
                    "Feature importance shows which variables have the most influence on predictions. "
                    "This helps with interpretability and feature selection."
                ))

                top_n = min(10, len(features))
                top_features = features[:top_n]
                names = [f[0] for f in top_features]
                scores_imp = [f[1] for f in top_features]

                fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.4)))
                colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
                bars = ax.barh(range(len(names)), scores_imp, color=colors)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'Feature Importance — {model_name}', fontsize=14, fontweight='bold')
                ax.invert_yaxis()

                for bar, score in zip(bars, scores_imp):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                            f'{score:.4f}', va='center', fontsize=9)

                plt.tight_layout()
                img_b64 = plot_to_base64(fig)

                cells.append(create_code_cell(
                    """# Feature importance — which variables matter most?
import matplotlib.pyplot as plt
import numpy as np

try:
    fi_df = predictor.feature_importance(X_test, silent=True)
    if fi_df is not None and not fi_df.empty:
        fi_df = fi_df.sort_values('importance', ascending=False).head(10)
        names = fi_df.index.tolist()
        scores = [float(fi_df.loc[n, 'importance']) for n in names]

        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.4)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
        bars = ax.barh(range(len(names)), scores, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance', fontsize=14)
        ax.invert_yaxis()
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{score:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importance not available.")
except Exception as e:
    print(f"Could not compute feature importance: {e}")""",
                    [create_image_output(img_b64)]
                ))

    return cells


def generate_recommendation_section(recommendations: dict) -> list:
    """Generate cells for the recommendations section — fully executable."""
    cells = []

    cells.append(create_markdown_cell(
        "## 💡 Step 6: Recommendations & Insights\n\n"
        "Based on the model evaluation results, here are our recommendations and insights."
    ))

    # --- EXECUTABLE: Model summary table ---
    if 'model_summary' in recommendations:
        summary_df = pd.DataFrame(recommendations['model_summary'])
        summary_html = summary_df.to_html(classes='table table-striped', border=0, index=False)

        cells.append(create_markdown_cell("### 📋 Model Performance Summary"))
        cells.append(create_code_cell(
            """# Model performance summary
import pandas as pd

# Build summary from leaderboard
score_cols = [c for c in leaderboard.columns if c not in [
    'model', 'score_val', 'pred_time_val', 'fit_time', 'pred_time_test',
    'fit_time_marginal', 'pred_time_val_marginal', 'pred_time_test_marginal',
    'stack_level', 'can_infer', 'fit_order']]
score_name = score_cols[0] if score_cols else 'score_test'

summary = leaderboard[['model', score_name]].head(10).copy()
summary.columns = ['Model', 'Score']
summary['Score'] = summary['Score'].round(4)
summary""",
            [{
                'output_type': 'execute_result',
                'data': {
                    'text/html': [summary_html],
                    'text/plain': [summary_df.to_string()]
                },
                'metadata': {},
                'execution_count': None
            }]
        ))

    # --- EXECUTABLE: Best model & recommendations ---
    if 'overall_verdict' in recommendations:
        cells.append(create_markdown_cell(
            f"### 🏆 Overall Verdict\n\n{recommendations['overall_verdict']}"
        ))

    # --- EXECUTABLE: Generate recommendations inline ---
    if 'nl_recommendations' in recommendations:
        cells.append(create_markdown_cell(
            "### 🎯 Actionable Recommendations\n\n"
            + "\n\n".join(recommendations['nl_recommendations'])
        ))

    # Add a code cell that generates equivalent recommendations
    best_model_name = recommendations.get('best_model', 'N/A')
    best_score = recommendations.get('best_score', 0)
    cells.append(create_code_cell(
        f"""# Generate actionable recommendations based on model performance
import numpy as np

best_model = predictor.model_best
score_cols = [c for c in leaderboard.columns if c not in [
    'model', 'score_val', 'pred_time_val', 'fit_time', 'pred_time_test',
    'fit_time_marginal', 'pred_time_val_marginal', 'pred_time_test_marginal',
    'stack_level', 'can_infer', 'fit_order']]
score_name = score_cols[0] if score_cols else 'score_test'
best_score = float(leaderboard[leaderboard['model'] == best_model][score_name].iloc[0])

print(f"🏆 Best Model: {{best_model}} (Score: {{best_score:.4f}})")

if best_score < 0.7:
    print(f"⚠️ Performance Warning: Score is {{best_score:.4f}}, below 0.7.")
    print("   - Try a higher quality preset (QUALITY_PRESET = 'best_quality')")
    print("   - Collect more training data")
    print("   - Check for data leakage or label errors")
elif best_score < 0.9:
    print(f"👍 Good performance ({{best_score:.4f}}). To improve further:")
    print("   - Try QUALITY_PRESET = 'best_quality'")
    print("   - Apply feature engineering on top predictors")
else:
    print(f"🌟 Excellent performance ({{best_score:.4f}})! Production-ready.")

# Show top features if available
try:
    fi_df = predictor.feature_importance(X_test, silent=True)
    if fi_df is not None and not fi_df.empty:
        top = fi_df.sort_values('importance', ascending=False).head(3)
        print(f"\\n🔑 Top features: {{', '.join(top.index.tolist())}}")
except Exception:
    pass""",
        [create_text_output(
            f"🏆 Best Model: {best_model_name} (Score: {best_score:.4f})\n"
            + ("👍 Good performance." if best_score >= 0.7 else "⚠️ Needs improvement.")
        )]
    ))

    return cells


def find_best_model_local(all_metrics: dict, problem_type: str) -> tuple:
    """Find best model locally (not using prediction module)."""
    best_name = None
    best_score = -np.inf

    for name, metrics in all_metrics.items():
        if problem_type in ('classification', 'binary', 'multiclass'):
            score = metrics.get('accuracy', 0)
        else:
            score = metrics.get('r2', -np.inf)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, all_metrics.get(best_name, {})


def generate_notebook(df_clean, cleaning_steps, info, split_data, training_results,
                       cv_results, eval_results, recommendations, output_path: str,
                       dataset_name: str = 'Dataset',
                       csv_path_for_nb: str = '../input/dataset.csv',
                       target_col: str = None,
                       problem_type: str = None,
                       quality_preset: str = 'medium_quality_faster_train',
                       time_limit: int = None,
                       seed: int = 42,
                       test_size: float = 0.2) -> str:
    """
    Generate the complete Jupyter Notebook and save it to output_path.

    The notebook contains fully executable Python code in every cell,
    with pre-computed outputs embedded for instant visual feedback.
    """
    cells = []

    # ========== CONFIG HEADER (fully executable) ==========
    cells.extend(make_config_header(
        csv_path=csv_path_for_nb,
        target_col=target_col,
        problem_type=problem_type,
        quality_preset=quality_preset,
        time_limit=time_limit,
        seed=seed,
        test_size=test_size,
    ))

    # ========== TITLE ==========
    cells.append(create_markdown_cell(
        f"# 📓 BlackBox AutoML Report — {dataset_name}\n\n"
        "This notebook was **automatically generated** by BlackBox AutoML. "
        "It contains the complete machine learning pipeline: "
        "from data loading and cleaning through model evaluation and recommendations.\n\n"
        "To re-run the full pipeline on a different dataset: change `CSV_PATH` in the cell above.\n\n"
        "---"
    ))

    # ========== SECTION 1: DATA CLEANING ==========
    cells.extend(generate_cleaning_section(df_clean, cleaning_steps, info, csv_path_for_nb))

    # ========== SECTION 2: EDA ==========
    cells.extend(generate_eda_section(df_clean, info))

    # ========== SECTION 3: DATA SPLITTING ==========
    cells.extend(generate_split_section(split_data))

    # ========== SECTION 4: TRAINING ==========
    cells.extend(generate_training_section(training_results, cv_results, split_data,
                                            quality_preset, time_limit))

    # ========== SECTION 5: EVALUATION ==========
    cells.extend(generate_evaluation_section(eval_results, split_data))

    # ========== SECTION 6: RECOMMENDATIONS ==========
    cells.extend(generate_recommendation_section(recommendations))

    # ========== FOOTER ==========
    cells.append(create_markdown_cell(
        "---\n"
        "*Report generated by **BlackBox AutoML** — Automated Machine Learning Pipeline*\n\n"
        "_Thank you for using BlackBox AutoML!_"
    ))

    # Load template and build notebook
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'templates', 'report_template.ipynb.json')
    with open(template_path, 'r') as f:
        notebook = json.load(f)

    notebook['cells'] = cells

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    return output_path