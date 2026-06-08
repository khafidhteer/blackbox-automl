"""
recommendation.py
Generates recommendations and insights based on model evaluation results
from AutoGluon's leaderboard and evaluation output.
"""

import numpy as np


def generate_recommendations(all_metrics: dict, problem_type: str,
                              leaderboard=None, cv_results: dict = None,
                              feature_importance: dict = None,
                              dataset_info: dict = None) -> dict:
    """
    Generate comprehensive recommendations based on model performance
    from AutoGluon evaluation.

    Parameters
    ----------
    all_metrics : dict
        Dictionary mapping model_name -> metrics dict (from prediction.py).
    problem_type : str
        'classification' or 'regression'.
    leaderboard : pd.DataFrame, optional
        AutoGluon leaderboard with all model scores.
    cv_results : dict, optional
        Kept for backward compatibility.
    feature_importance : dict, optional
        Mapping model_name -> list of (feature, importance) tuples.
    dataset_info : dict, optional
        Additional dataset metadata.

    Returns
    -------
    dict containing recommendations, model summary, and overall verdict.
    """
    recommendations = {}

    if not all_metrics:
        return {'error': 'No model metrics available for recommendations.'}

    # Find best model
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

    recommendations['best_model'] = best_model_name
    recommendations['best_score'] = round(best_score, 4)

    # Model comparison summary
    model_summary = []
    for name, metrics in all_metrics.items():
        if problem_type == 'classification':
            summary = {
                'model': name,
                'accuracy': round(metrics.get('accuracy', 0), 4),
                'precision': round(metrics.get('precision', 0), 4),
                'recall': round(metrics.get('recall', 0), 4),
                'f1_score': round(metrics.get('f1_score', 0), 4),
                'roc_auc': round(metrics.get('roc_auc', 0), 4) if 'roc_auc' in metrics else None
            }
        else:
            summary = {
                'model': name,
                'r2': round(metrics.get('r2', 0), 4),
                'mae': round(metrics.get('mae', 0), 4),
                'rmse': round(metrics.get('rmse', 0), 4),
                'mse': round(metrics.get('mse', 0), 4)
            }

        # Add AutoGluon leaderboard scores if available
        if leaderboard is not None:
            lb_row = leaderboard[leaderboard['model'] == name]
            if not lb_row.empty:
                # Find the score column (not model metadata)
                for col in lb_row.columns:
                    if col not in ['model', 'score_val', 'pred_time_val',
                                   'fit_time', 'pred_time_test',
                                   'fit_time_marginal', 'pred_time_val_marginal',
                                   'pred_time_test_marginal', 'stack_level',
                                   'can_infer', 'fit_order']:
                        summary['leaderboard_score'] = round(
                            float(lb_row[col].iloc[0]), 4
                        )
                        break

                # Add timing info
                pred_time = lb_row['pred_time_test'].iloc[0]
                if pred_time and not np.isnan(pred_time):
                    summary['prediction_time_sec'] = round(float(pred_time), 4)

        # Add cross-validation info (from legacy cv_results)
        if cv_results and name in cv_results:
            cv = cv_results[name]
            if cv.get('cv_scores') is not None:
                summary['cv_mean'] = round(cv['cv_mean'], 4)
                summary['cv_std'] = round(cv['cv_std'], 4)

        model_summary.append(summary)

    recommendations['model_summary'] = model_summary

    # Generate natural language recommendations
    nl_recommendations = []
    nl_recommendations.append(
        f"🎯 **Best Model**: {best_model_name} achieved the best performance with "
        f"{'accuracy' if problem_type == 'classification' else 'R² score'} of {best_score:.4f}."
    )

    # Explain AutoGluon's cross-validation / leaderboard scores
    if leaderboard is not None:
        best_lb = leaderboard[leaderboard['model'] == best_model_name]
        if not best_lb.empty:
            score_val = best_lb['score_val'].iloc[0]
            if score_val and not np.isnan(score_val):
                nl_recommendations.append(
                    f"📊 **Validation Score**: {best_model_name} achieved a validation score of "
                    f"{score_val:.4f}, indicating "
                    f"{'stable' if abs(score_val - best_score) < 0.05 else 'somewhat variable'} "
                    f"performance between validation and test sets."
                )

    # Feature importance recommendations
    if feature_importance and best_model_name in feature_importance:
        top_features = feature_importance[best_model_name][:5]
        nl_recommendations.append(
            f"🔑 **Top Features**: The most important features for {best_model_name} are: "
            + ", ".join([f"{f[0]} ({f[1]:.4f})" for f in top_features])
        )

        if len(top_features) > 0:
            nl_recommendations.append(
                f"💡 **Feature Engineering Tip**: Consider creating interaction terms or polynomial "
                f"features from the top predictors to potentially improve model performance."
            )

    # Performance improvement suggestions
    if best_score < 0.7:
        nl_recommendations.append(
            f"⚠️ **Performance Warning**: The best model's {'accuracy' if problem_type == 'classification' else 'R² score'} "
            f"is {best_score:.4f}, which is below 0.7. Consider:"
        )
        nl_recommendations.append(f"   - Collecting more training data")
        nl_recommendations.append(f"   - Running with --quality balanced for medium-quality models")
        nl_recommendations.append(f"   - Running with --quality best for full AutoGluon (neural networks, stacking)")
        nl_recommendations.append(f"   - Checking for data leakage or label errors")
    elif best_score < 0.9:
        nl_recommendations.append(
            f"👍 **Good Performance**: The model shows {'good' if best_score >= 0.8 else 'moderate'} "
            f"performance. To further improve:"
        )
        nl_recommendations.append(f"   - Try --quality balanced or --quality best for better models")
        nl_recommendations.append(f"   - Apply feature engineering on top predictors")
        nl_recommendations.append(f"   - Collect more training data if feasible")
    else:
        nl_recommendations.append(
            f"🌟 **Excellent Performance**: The model achieves excellent results! "
            f"Consider the following for production deployment:"
        )
        nl_recommendations.append(f"   - Validate on a separate holdout dataset")
        nl_recommendations.append(f"   - Monitor for data drift over time")
        nl_recommendations.append(f"   - Consider model compression for faster inference")

    # Problem-specific recommendations
    if problem_type == 'classification':
        # Check class balance via classification_report
        for name, metrics in all_metrics.items():
            if 'classification_report' in metrics:
                report = metrics['classification_report']
                if 'weighted avg' in report:
                    avg = report['weighted avg']
                    if avg.get('recall', 1) < 0.7 or avg.get('precision', 1) < 0.7:
                        nl_recommendations.append(
                            f"⚖️ **Class Imbalance**: Weighted precision/recall is below 0.7. "
                            f"Consider using SMOTE, class weights, or collecting more samples for minority classes."
                        )
                        break
    else:
        # Regression-specific: check for residual bias
        for name, metrics in all_metrics.items():
            residuals = np.array(metrics.get('residuals', []))
            if len(residuals) > 0:
                residual_std = np.std(residuals)
                residual_mean = np.mean(residuals)
                if residual_std > 0 and abs(residual_mean) > 0.1 * residual_std:
                    nl_recommendations.append(
                        f"📈 **Residual Bias**: The mean residual ({residual_mean:.4f}) suggests possible bias. "
                        f"Consider transforming the target variable (log, Box-Cox) or trying higher quality settings."
                    )
                    break

    recommendations['nl_recommendations'] = nl_recommendations
    recommendations['overall_verdict'] = _get_overall_verdict(best_score, problem_type)

    return recommendations


def _get_overall_verdict(best_score: float, problem_type: str) -> str:
    """Generate an overall verdict string."""
    metric_name = 'accuracy' if problem_type == 'classification' else 'R²'

    if best_score >= 0.95:
        return (f"🚀 **Outstanding!** The model achieves {metric_name} of {best_score:.4f}. "
                "This model is production-ready with minimal improvements needed.")
    elif best_score >= 0.85:
        return (f"✅ **Great!** The model achieves {metric_name} of {best_score:.4f}. "
                "This model is suitable for most use cases with some fine-tuning.")
    elif best_score >= 0.75:
        return (f"👍 **Good.** The model achieves {metric_name} of {best_score:.4f}. "
                "It's a solid baseline but improvements can be made.")
    elif best_score >= 0.6:
        return (f"⚠️ **Fair.** The model achieves {metric_name} of {best_score:.4f}. "
                "More work is needed for production deployment.")
    else:
        return (f"❌ **Poor.** The model only achieves {metric_name} of {best_score:.4f}. "
                "Significant improvements needed in data quality, feature engineering, or modeling approach.")


def run_pipeline(all_metrics: dict, problem_type: str,
                 leaderboard=None, cv_results: dict = None,
                 feature_importance: dict = None,
                 dataset_info: dict = None) -> dict:
    """Run the recommendation pipeline."""
    return generate_recommendations(
        all_metrics, problem_type,
        leaderboard=leaderboard,
        cv_results=cv_results,
        feature_importance=feature_importance,
        dataset_info=dataset_info
    )