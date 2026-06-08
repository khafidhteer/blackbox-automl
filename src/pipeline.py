"""
pipeline.py
Main orchestrator for BlackBox AutoML — AutoGluon Edition.
Runs the complete pipeline: load → clean → split → train (AutoGluon) → evaluate → recommend → generate notebook.
"""

import os
import sys
import argparse
import glob
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_cleaning import load_csv, run_pipeline as clean_pipeline
from src.data_split import run_pipeline as split_pipeline
from src.training import run_pipeline as train_pipeline, cleanup as cleanup_autogluon
from src.prediction import run_pipeline as eval_pipeline
from src.recommendation import run_pipeline as recommend_pipeline
from src.notebook_generator import generate_notebook


def find_csv_file(input_dir: str = 'input') -> str:
    """Find a CSV file in the input directory."""
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        return None

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    # Also check root directory
    root_csv_files = glob.glob(os.path.join(os.path.dirname(input_dir), '*.csv'))
    csv_files.extend(root_csv_files)

    if not csv_files:
        return None

    # Return the most recently modified file
    return max(csv_files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description='BlackBox AutoML - Automated Machine Learning Pipeline (AutoGluon)'
    )
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input CSV file. If not provided, scans input/ folder.')
    parser.add_argument('--output', '-o', type=str, default='output/automl_report.ipynb',
                        help='Path for the output notebook.')
    parser.add_argument('--target', '-t', type=str, default=None,
                        help='Target column name. Auto-detected if not provided.')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size ratio (default: 0.2)')
    parser.add_argument('--quality', '-q', type=str, default='lightweight',
                        choices=['lightweight', 'balanced', 'best'],
                        help='Model quality preset: lightweight (fast, default), balanced, best (full AutoGluon)')
    parser.add_argument('--time-limit', type=int, default=None,
                        help='Time limit for AutoGluon training in seconds (default: no limit)')
    parser.add_argument('--missing-strategy', type=str, default='auto',
                        choices=['auto', 'mean', 'median', 'mode', 'drop'],
                        help='Strategy for handling missing values')
    parser.add_argument('--outlier-method', type=str, default='cap',
                        choices=['cap', 'remove', 'none'],
                        help='Method for handling outliers')

    args = parser.parse_args()

    # ========== STEP 0: Find input file ==========
    print("=" * 60)
    print("  🤖 BlackBox AutoML Pipeline (AutoGluon)")
    print("=" * 60)

    if args.input:
        csv_path = args.input
        if not os.path.exists(csv_path):
            print(f"❌ Error: Input file '{csv_path}' not found.")
            sys.exit(1)
    else:
        csv_path = find_csv_file('input')
        if csv_path is None:
            print("❌ Error: No CSV file found in 'input/' folder.")
            print("   Please place a CSV file in the 'input/' directory and try again.")
            print("   Usage: python src/pipeline.py --input path/to/your/file.csv")
            sys.exit(1)

    print(f"\n📂 Input file: {csv_path}")
    print(f"⚙️  Quality preset: {args.quality}")
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ========== STEP 1: Load and Clean Data ==========
    print("\n" + "─" * 40)
    print("  Step 1: Data Loading & Cleaning")
    print("─" * 40)

    df = load_csv(csv_path)
    print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    cleaning_config = {
        'missing_strategy': args.missing_strategy,
        'remove_duplicates': True,
        'outlier_method': args.outlier_method
    }
    df_clean, cleaning_steps, info = clean_pipeline(df, cleaning_config)

    print(f"   After cleaning: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    for step in cleaning_steps:
        print(f"   ✓ {step}")

    # ========== STEP 2: Split Data ==========
    print("\n" + "─" * 40)
    print("  Step 2: Data Splitting")
    print("─" * 40)

    split_data = split_pipeline(df_clean, target_col=args.target, test_size=args.test_size)
    print(f"   Target column: {split_data['target_col']}")
    print(f"   Problem type: {split_data['problem_type']}")
    print(f"   Training set: {split_data['X_train'].shape}")
    print(f"   Testing set: {split_data['X_test'].shape}")

    # ========== STEP 3: Train Models (AutoGluon) ==========
    print("\n" + "─" * 40)
    print("  Step 3: AutoGluon Training")
    print("─" * 40)

    print(f"   Quality: {args.quality}")
    if args.time_limit:
        print(f"   Time limit: {args.time_limit}s")
    print(f"   Training models (lightweight-first strategy)...")

    train_results = train_pipeline(
        split_data['X_train'], split_data['y_train'],
        split_data['X_test'], split_data['y_test'],
        split_data['problem_type'],
        quality=args.quality,
        time_limit=args.time_limit
    )

    # Display number of models trained
    lb = train_results['leaderboard']
    num_models = len(lb)
    print(f"   ✅ AutoGluon trained {num_models} models")
    print(f"   🏆 Best model (AutoGluon): {train_results['model_name']}")

    # Show top models from leaderboard
    print(f"\n   📊 Leaderboard (top 5):")
    score_cols = [c for c in lb.columns if c not in [
        'model', 'score_val', 'pred_time_val', 'fit_time',
        'pred_time_test', 'fit_time_marginal', 'pred_time_val_marginal',
        'pred_time_test_marginal', 'stack_level', 'can_infer', 'fit_order'
    ]]
    score_name = score_cols[0] if score_cols else 'score_test'
    for idx, row in lb.head(5).iterrows():
        score_val = row.get(score_name, row.get('score_test', 'N/A'))
        print(f"      {idx+1}. {row['model']}: {score_val:.4f}")

    # ========== STEP 4: Evaluate Models ==========
    print("\n" + "─" * 40)
    print("  Step 4: Model Evaluation")
    print("─" * 40)

    feature_names = list(split_data['X_encoded'].columns)
    eval_results = eval_pipeline(
        train_results['predictor'],
        split_data['X_test'],
        split_data['y_test'],
        split_data['problem_type'],
        feature_names=feature_names,
        best_model_hint=train_results['model_name']
    )

    best_model_name = eval_results.get('best_model_name')
    if best_model_name and best_model_name in eval_results.get('all_metrics', {}):
        best_metrics = eval_results['all_metrics'][best_model_name]
        score = best_metrics.get('accuracy' if split_data['problem_type'] == 'classification' else 'r2', 0)
        print(f"   🏆 Best model: {best_model_name} (score: {score:.4f})")
    else:
        print("   ⚠️ No models were successfully evaluated.")

    # ========== STEP 5: Generate Recommendations ==========
    print("\n" + "─" * 40)
    print("  Step 5: Generating Recommendations")
    print("─" * 40)

    recommendations = recommend_pipeline(
        eval_results['all_metrics'],
        split_data['problem_type'],
        leaderboard=train_results['leaderboard'],
        feature_importance=eval_results['feature_importance']
    )

    if 'overall_verdict' in recommendations:
        print(f"   {recommendations['overall_verdict'][:80]}...")

    # ========== STEP 6: Generate Notebook ==========
    print("\n" + "─" * 40)
    print("  Step 6: Generating Notebook")
    print("─" * 40)

    output_path = generate_notebook(
        df_clean=df_clean,
        cleaning_steps=cleaning_steps,
        info=info,
        split_data=split_data,
        training_results=train_results,
        cv_results={},  # AutoGluon handles CV internally; we use leaderboard instead
        eval_results=eval_results,
        recommendations=recommendations,
        output_path=args.output,
        dataset_name=dataset_name
    )

    # ========== CLEANUP: Remove AutoGluon temp model dir ==========
    cleanup_autogluon(train_results.get('predictor'))

    print(f"\n{'=' * 60}")
    print(f"  ✅ Pipeline Complete!")
    print(f"  📓 Notebook saved to: {output_path}")
    print(f"{'=' * 60}")
    print(f"\nOpen the notebook with:")
    print(f"   jupyter notebook {output_path}")
    print(f"   # or open in VS Code / your preferred viewer")


if __name__ == '__main__':
    main()