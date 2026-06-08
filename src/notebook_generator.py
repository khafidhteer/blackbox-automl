"""
notebook_generator.py
Generates the final Jupyter Notebook (.ipynb) with all steps explained.
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


def generate_cleaning_section(df_clean, cleaning_steps, info) -> list:
    """Generate cells for the data cleaning section."""
    cells = []

    # Title
    cells.append(create_markdown_cell(
        "## 📋 Step 1: Data Loading & Cleansing\n\n"
        "In this section, we load the dataset and perform initial data cleaning. "
        "Cleaning is a critical step in any machine learning pipeline — "
        "it ensures the data is consistent, complete, and ready for modeling."
    ))

    # Load data code
    code = """# Load the dataset
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Display basic information
print(f"Dataset shape: {info['shape']}")
print(f"Number of columns: {info['shape'][1]}")
print(f"Number of rows: {info['shape'][0]}")
print(f"Number of duplicates: {info['duplicate_count']}")
print(f"\nColumns and their data types:")
for col, dtype in info['dtypes'].items():
    print(f"  - {col}: {dtype}")""".replace("info['shape']", str(info['shape'])) \
        .replace("info['shape'][1]", str(info['shape'][1])) \
        .replace("info['shape'][0]", str(info['shape'][0])) \
        .replace("info['duplicate_count']", str(info['duplicate_count']))

    cells.append(create_code_cell(code, [
        create_text_output(
            f"Dataset shape: {info['shape']}\n"
            f"Number of columns: {info['shape'][1]}\n"
            f"Number of rows: {info['shape'][0]}\n"
            f"Number of duplicates: {info['duplicate_count']}\n\n"
            f"Columns and their data types:\n"
            + "\n".join([f"  - {col}: {dtype}" for col, dtype in info['dtypes'].items()])
        )
    ]))

    # Display first few rows
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

    # Handling missing values
    cells.append(create_markdown_cell(
        "### 🧹 Handling Missing Values\n\n"
        "Missing values can cause issues with many machine learning algorithms. "
        "We handle them using appropriate strategies:\n"
        "- **Numerical columns**: fill with median (robust to outliers)\n"
        "- **Categorical columns**: fill with mode (most frequent value)"
    ))

    # Missing values steps
    missing_steps_text = "\n".join([f"- {step}" for step in cleaning_steps if 'missing' in step.lower() or 'fill' in step.lower()])
    if missing_steps_text:
        cells.append(create_markdown_cell(
            f"**Missing Value Handling Results:**\n\n{missing_steps_text}"
        ))

    # Duplicates
    dup_steps = [s for s in cleaning_steps if 'duplicate' in s.lower()]
    if dup_steps:
        cells.append(create_markdown_cell(
            "### 🔄 Removing Duplicates\n\n"
            f"{dup_steps[0]}"
        ))

    # Outliers section
    outlier_steps = [s for s in cleaning_steps if 'outlier' in s.lower() or 'cap' in s.lower()]
    if outlier_steps:
        cells.append(create_markdown_cell(
            "### 📊 Outlier Detection & Treatment\n\n"
            "Outliers can skew the model's learning. We use the **IQR (Interquartile Range)** method "
            "to detect and cap extreme values.\n\n"
            "**Results:**\n" + "\n".join([f"- {s}" for s in outlier_steps])
        ))

    # Final info
    cells.append(create_markdown_cell(
        "### ✅ Data Cleaning Complete\n\n"
        f"The dataset now has **{df_clean.shape[0]} rows** and **{df_clean.shape[1]} columns**, "
        "ready for the next steps."
    ))

    return cells


def generate_eda_section(df_clean, info) -> list:
    """Generate cells for exploratory data analysis."""
    cells = []

    cells.append(create_markdown_cell(
        "## 📊 Step 2: Exploratory Data Analysis (EDA)\n\n"
        "EDA helps us understand patterns, relationships, and distributions in the data. "
        "We'll visualize key aspects to gain insights."
    ))

    # Statistical summary
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

    # Correlation heatmap
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
            "# Correlation heatmap\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
            "fig, ax = plt.subplots(figsize=(12, 8))\n"
            f"corr = df[numeric_cols].corr()\n"
            "mask = np.triu(np.ones_like(corr, dtype=bool), k=1)\n"
            "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax)\n"
            "plt.title('Feature Correlation Matrix', fontsize=14)\n"
            "plt.tight_layout()\n"
            "plt.show()".replace("numeric_cols", str(list(numeric_cols))),
            [create_image_output(img_b64)]
        ))

    # Distribution plots for numeric columns
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

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        img_b64 = plot_to_base64(fig)

        cells.append(create_code_cell(
            "# Distribution histograms\n"
            "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n"
            "axes = axes.flatten()\n"
            "for i, col in enumerate(df.select_dtypes(include=[np.number]).columns[:6]):\\n"
            "    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=axes[i], color='steelblue')\\n"
            "    axes[i].set_title(f'Distribution of {col}')\\n"
            "plt.tight_layout()\\n"
            "plt.show()",
            [create_image_output(img_b64)]
        ))

    return cells


def generate_split_section(split_data: dict) -> list:
    """Generate cells for the data splitting section."""
    cells = []

    split_info = split_data['split_info']
    problem_type = split_data['problem_type']

    cells.append(create_markdown_cell(
        "## 🔀 Step 3: Data Splitting\n\n"
        "We split the data into:\n"
        "- **Features (X)**: The input variables used for prediction\n"
        "- **Target (y)**: The value we want to predict\n\n"
        f"**Problem Type Detected**: `{problem_type.upper()}`\n\n"
        "Then we split into training and testing sets to evaluate model performance on unseen data."
    ))

    cells.append(create_markdown_cell(
        f"### 🎯 Target Column\n\n"
        f"The target column is **`{split_data['target_col']}`**. "
        f"This is the variable we want to predict.\n\n"
        f"**Problem Type**: `{problem_type}` — we'll use "
        f"{'classification' if problem_type == 'classification' else 'regression'} models."
    ))

    # Target distribution
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

        # Add value labels on bars
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        plt.tight_layout()
        img_b64 = plot_to_base64(fig)

        cells.append(create_code_cell(
            f"# Target distribution\n"
            f"value_counts = y.value_counts()\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "ax.bar(value_counts.index.astype(str), value_counts.values, color=['#2E86AB', '#A23B72', '#F18F01'])\n"
            "ax.set_title('Target Class Distribution')\n"
            "ax.set_xlabel('Class')\n"
            "ax.set_ylabel('Count')\n"
            "plt.show()",
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
            f"# Target distribution (regression)\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            f"sns.histplot(y, kde=True, bins=30, color='steelblue', ax=ax)\n"
            "ax.set_title('Target Variable Distribution')\n"
            "plt.show()",
            [create_image_output(img_b64)]
        ))

    # Split info
    stratified_text = " (stratified)" if split_info['stratified'] else ""
    cells.append(create_markdown_cell(
        f"### ✂️ Train/Test Split\n\n"
        f"- **Training set**: {split_info['train_percentage']}% "
        f"({split_info['X_train_shape'][0]} samples, {split_info['X_train_shape'][1]} features)\n"
        f"- **Testing set**: {split_info['test_percentage']}% "
        f"({split_info['X_test_shape'][0]} samples, {split_info['X_test_shape'][1]} features)\n"
        f"- **Stratified**: {split_info['stratified']}\n\n"
        f"{'✅ Stratified split ensures class proportions are preserved in both sets.' if split_info['stratified'] else ''}"
    ))

    return cells


def generate_training_section(training_results: dict, cv_results: dict, split_data: dict) -> list:
    """Generate cells for the model training section."""
    cells = []
    problem_type = split_data['problem_type']

    # Determine if we're using AutoGluon (new) or manual (old) training results
    is_autogluon = 'predictor' in training_results or 'leaderboard' in training_results

    cells.append(create_markdown_cell(
        f"## 🧠 Step 4: Model Training\n\n"
        f"Training multiple {problem_type} models to find the best performer. "
        f"{'We use **AutoGluon** — an automated ML framework that handles model selection, '
         'hyperparameter tuning, ensembling, and cross-validation automatically.' if is_autogluon
         else 'We train each model on the training data and evaluate using cross-validation.'}"
    ))

    # Models trained
    if is_autogluon:
        lb = training_results.get('leaderboard')
        if lb is not None:
            num_models = len(lb)
            best_model = training_results.get('model_name', 'N/A')
            model_list = "\n".join([f"- {row['model']}" for _, row in lb.iterrows()])
            cells.append(create_markdown_cell(
                "### 🤖 Models Trained (AutoGluon)\n\n"
                f"A total of **{num_models} models** were trained:\n\n"
                f"{model_list}\n\n"
                f"🏆 **Best Model**: `{best_model}`"
            ))

        # Display AutoGluon Leaderboard
        if lb is not None:
            cells.append(create_markdown_cell(
                "### 📊 AutoGluon Leaderboard\n\n"
                "The leaderboard shows all trained models ranked by performance. "
                "AutoGluon automatically applies cross-validation and ranks models "
                "by their validation score."
            ))

            # Format leaderboard for display
            display_cols = [c for c in lb.columns
                            if c not in ['fit_time_marginal', 'pred_time_val_marginal',
                                         'pred_time_test_marginal', 'stack_level',
                                         'can_infer', 'fit_order']]
            lb_display = lb[display_cols].round(4) if len(display_cols) > 0 else lb.round(4)

            lb_html = lb_display.to_html(classes='table table-striped', border=0, index=False)
            cells.append(create_code_cell(
                "# AutoGluon leaderboard — all model results\n"
                "import pandas as pd\n"
                "leaderboard = predictor.leaderboard(test_data, silent=True)\n"
                "leaderboard",
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

            # Leaderboard bar chart (top models)
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
                "# Leaderboard visualization\n"
                "fig, ax = plt.subplots(figsize=(10, 6))\n"
                f"models = {model_names}\n"
                f"scores = {scores}\n"
                "ax.barh(range(len(models)), scores, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(models))))\n"
                "ax.set_yticks(range(len(models)))\n"
                "ax.set_yticklabels(models)\n"
                "ax.set_xlabel('Test Score')\n"
                "ax.set_title('Model Leaderboard')\n"
                "plt.show()",
                [create_image_output(img_b64)]
            ))

    else:
        # Legacy: manual training results (list of dicts)
        if isinstance(training_results, list):
            cells.append(create_markdown_cell(
                "### 🤖 Models Trained\n\n"
                "The following models were trained:\n"
                + "\n".join([f"- {r['model_name']}" for r in training_results if r.get('success')])
            ))

        # Cross-validation results table
        if cv_results:
            cells.append(create_markdown_cell(
                "### 📊 Cross-Validation Results\n\n"
                "Cross-validation provides a more robust estimate of model performance by "
                "training and evaluating on multiple data splits."
            ))

            cv_table_data = []
            for name, cv in cv_results.items():
                if cv.get('cv_scores') is not None:
                    cv_table_data.append({
                        'Model': name,
                        'CV Mean': f"{cv['cv_mean']:.4f}",
                        'CV Std': f"{cv['cv_std']:.4f}",
                        'CV Min': f"{cv['cv_min']:.4f}",
                        'CV Max': f"{cv['cv_max']:.4f}"
                    })

            if cv_table_data:
                cv_df = pd.DataFrame(cv_table_data)
                cv_html = cv_df.to_html(classes='table table-striped', border=0, index=False)
                cells.append(create_code_cell(
                    "# Cross-validation results summary\n"
                    "import pandas as pd\n"
                    "# Results summarized below",
                    [{
                        'output_type': 'execute_result',
                        'data': {
                            'text/html': [cv_html],
                            'text/plain': [cv_df.to_string()]
                        },
                        'metadata': {},
                        'execution_count': None
                    }]
                ))

                # CV comparison chart
                fig, ax = plt.subplots(figsize=(10, 5))
                cv_models = [d['Model'] for d in cv_table_data]
                means = [float(d['CV Mean']) for d in cv_table_data]
                stds = [float(d['CV Std']) for d in cv_table_data]

                colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cv_models)))
                bars = ax.barh(range(len(cv_models)), means, xerr=stds, color=colors, capsize=5)
                ax.set_yticks(range(len(cv_models)))
                ax.set_yticklabels(cv_models)
                ax.set_xlabel('Cross-Validation Score')
                ax.set_title(f'Model Comparison — {problem_type.title()}', fontsize=14, fontweight='bold')
                ax.set_xlim(0, min(1.1, max(means) * 1.3))

                for bar, mean, std in zip(bars, means, stds):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{mean:.4f} ± {std:.4f}', va='center', fontsize=10)

                plt.tight_layout()
                img_b64 = plot_to_base64(fig)

                cells.append(create_code_cell(
                    "# Cross-validation comparison\n"
                    "fig, ax = plt.subplots(figsize=(10, 5))\n"
                    f"models = {cv_models}\\n"
                    f"means = {means}\\n"
                    f"stds = {stds}\\n"
                    "ax.barh(range(len(models)), means, xerr=stds, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(models))))\\n"
                    "ax.set_yticks(range(len(models)))\\n"
                    "ax.set_yticklabels(models)\\n"
                    "ax.set_xlabel('CV Score')\\n"
                    "plt.show()",
                    [create_image_output(img_b64)]
                ))

    return cells


def generate_evaluation_section(eval_results: dict, split_data: dict) -> list:
    """Generate cells for the evaluation section."""
    cells = []
    all_metrics = eval_results['all_metrics']
    problem_type = split_data['problem_type']

    cells.append(create_markdown_cell(
        "## 📈 Step 5: Model Evaluation\n\n"
        "Evaluating each model on the held-out test set to compare real-world performance."
    ))

    if problem_type == 'classification':
        # Metrics table
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
            "# Classification metrics comparison\n"
            "import pandas as pd\n"
            "# Results summarized below",
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

        # Confusion matrices
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
                "# Confusion matrix\n"
                "fig, ax = plt.subplots(figsize=(8, 6))\n"
                f"cm = {cm.tolist()}\\n"
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)\\n"
                "ax.set_title(f'Confusion Matrix — Best Model')\\n"
                "ax.set_xlabel('Predicted')\\n"
                "ax.set_ylabel('True')\\n"
                "plt.show()",
                [create_image_output(img_b64)]
            ))

    else:
        # Regression metrics
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
            "# Regression metrics comparison\n"
            "import pandas as pd\n"
            "# Results summarized below",
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

        # Residual plot for best model
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

            # Residuals vs Predicted
            axes[0].scatter(y_pred, residuals, alpha=0.6, color='steelblue')
            axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
            axes[0].set_xlabel('Predicted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')

            # Histogram of residuals
            axes[1].hist(residuals, bins=30, edgecolor='white', color='steelblue', alpha=0.7)
            axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1)
            axes[1].set_xlabel('Residuals')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')

            plt.tight_layout()
            img_b64 = plot_to_base64(fig)

            cells.append(create_code_cell(
                "# Residual plot\n"
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
                "axes[0].scatter(y_pred, residuals, alpha=0.6)\n"
                "axes[0].axhline(y=0, color='red', linestyle='--')\n"
                "axes[0].set_xlabel('Predicted')\n"
                "axes[0].set_ylabel('Residuals')\n"
                "axes[1].hist(residuals, bins=30, edgecolor='white')\n"
                "axes[1].set_xlabel('Residuals')\n"
                "axes[1].set_ylabel('Frequency')\n"
                "plt.show()",
                [create_image_output(img_b64)]
            ))

    # Feature importance
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
                scores = [f[1] for f in top_features]

                fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.4)))
                colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
                bars = ax.barh(range(len(names)), scores, color=colors)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'Feature Importance — {model_name}', fontsize=14, fontweight='bold')
                ax.invert_yaxis()

                for bar, score in zip(bars, scores):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                            f'{score:.4f}', va='center', fontsize=9)

                plt.tight_layout()
                img_b64 = plot_to_base64(fig)

                cells.append(create_code_cell(
                    "# Feature importance plot\n"
                    "fig, ax = plt.subplots(figsize=(10, 6))\n"
                    f"features = {top_features}\n"
                    "names = [f[0] for f in features]\n"
                    "scores = [f[1] for f in features]\n"
                    "ax.barh(range(len(names)), scores, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(names))))\n"
                    "ax.set_yticks(range(len(names)))\n"
                    "ax.set_yticklabels(names)\n"
                    "ax.set_xlabel('Importance')\n"
                    "ax.set_title('Feature Importance')\n"
                    "plt.show()",
                    [create_image_output(img_b64)]
                ))

    return cells


def generate_recommendation_section(recommendations: dict) -> list:
    """Generate cells for the recommendations section."""
    cells = []

    cells.append(create_markdown_cell(
        "## 💡 Step 6: Recommendations & Insights\n\n"
        "Based on the model evaluation results, here are our recommendations and insights."
    ))

    # Model summary table
    if 'model_summary' in recommendations:
        summary_df = pd.DataFrame(recommendations['model_summary'])
        summary_html = summary_df.to_html(classes='table table-striped', border=0, index=False)
        cells.append(create_markdown_cell("### 📋 Model Performance Summary"))
        cells.append(create_code_cell(
            "# Model performance summary\n"
            "# Results displayed below",
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

    # Overall verdict
    if 'overall_verdict' in recommendations:
        cells.append(create_markdown_cell(
            f"### 🏆 Overall Verdict\n\n{recommendations['overall_verdict']}"
        ))

    # Natural language recommendations
    if 'nl_recommendations' in recommendations:
        nl_text = "### 🎯 Actionable Recommendations\n\n"
        nl_text += "\n\n".join(recommendations['nl_recommendations'])
        cells.append(create_markdown_cell(nl_text))

    return cells


def find_best_model_local(all_metrics: dict, problem_type: str) -> tuple:
    """Find best model locally (not using prediction module)."""
    best_name = None
    best_score = -np.inf

    for name, metrics in all_metrics.items():
        if problem_type == 'classification':
            score = metrics.get('accuracy', 0)
        else:
            score = metrics.get('r2', -np.inf)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, all_metrics.get(best_name, {})


def generate_notebook(df_clean, cleaning_steps, info, split_data, training_results,
                       cv_results, eval_results, recommendations, output_path: str,
                       dataset_name: str = 'Dataset') -> str:
    """
    Generate the complete Jupyter Notebook and save it to output_path.
    """
    cells = []

    # ========== TITLE ==========
    cells.append(create_markdown_cell(
        f"# 📓 BlackBox AutoML Report — {dataset_name}\n\n"
        "This notebook was **automatically generated** by BlackBox AutoML. "
        "It contains the complete machine learning pipeline: "
        "from data loading and cleaning through model evaluation and recommendations.\n\n"
        "---"
    ))

    # ========== SECTION 1: DATA CLEANING ==========
    cells.extend(generate_cleaning_section(df_clean, cleaning_steps, info))

    # ========== SECTION 2: EDA ==========
    cells.extend(generate_eda_section(df_clean, info))

    # ========== SECTION 3: DATA SPLITTING ==========
    cells.extend(generate_split_section(split_data))

    # ========== SECTION 4: TRAINING ==========
    cells.extend(generate_training_section(training_results, cv_results, split_data))

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