"""
recommendation.py
Generates recommendations and insights based on model evaluation results.
"""

import numpy as np


def generate_recommendations(all_metrics: dict, problem_type: str, cv_results: dict = None,
                              feature_importance: dict = None, dataset_info: dict = None) -> dict:
    """
    Generate comprehensive recommendations based on model performance.
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

        # Add cross-validation info
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

    # Explain cross-validation scores
    if cv_results and best_model_name in cv_results:
        cv = cv_results[best_model_name]
        if cv.get('cv_scores') is not None:
            nl_recommendations.append(
                f"📊 **Cross-Validation**: {best_model_name} achieved a mean CV score of "
                f"{cv['cv_mean']:.4f} ± {cv['cv_std']:.4f}, indicating "
                f"{'stable' if cv['cv_std'] < 0.05 else 'somewhat variable'} performance across folds."
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
        nl_recommendations.append(f"   - Performing more extensive feature engineering")
        nl_recommendations.append(f"   - Trying advanced models like Neural Networks or Gradient Boosting")
        nl_recommendations.append(f"   - Checking for data leakage or label errors")
    elif best_score < 0.9:
        nl_recommendations.append(
            f"👍 **Good Performance**: The model shows {'good' if best_score >= 0.8 else 'moderate'} "
            f"performance. To further improve:"
        )
        nl_recommendations.append(f"   - Fine-tune hyperparameters with more extensive search")
        nl_recommendations.append(f"   - Try ensemble methods (stacking, voting)")
        nl_recommendations.append(f"   - Apply feature selection to remove noise")
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
        # Check class balance
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
        # Regression-specific
        for name, metrics in all_metrics.items():
            residuals = np.array(metrics.get('residuals', []))
            if len(residuals) > 0:
                residual_std = np.std(residuals)
                residual_mean = np.mean(residuals)
                if abs(residual_mean) > 0.1 * residual_std:
                    nl_recommendations.append(
                        f"📈 **Residual Bias**: The mean residual ({residual_mean:.4f}) suggests possible bias. "
                        f"Consider transforming the target variable (log, Box-Cox) or trying different model types."
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


def run_pipeline(all_metrics: dict, problem_type: str, cv_results: dict = None,
                  feature_importance: dict = None, dataset_info: dict = None) -> dict:
    """Run the recommendation pipeline."""
    return generate_recommendations(all_metrics, problem_type, cv_results, feature_importance, dataset_info)