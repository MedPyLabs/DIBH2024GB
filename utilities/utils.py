import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import calibration_curve
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
f1_score, roc_curve, auc)
from sklearn.utils import resample
import numpy as np

def save_feature_importances_to_csv(filename, model_name_prefix, top_models_path):
    # Collect feature importances for both calibrated and non-calibrated models
    loaded_top_models = joblib.load(top_models_path)
    calibrated_importances = []
    non_calibrated_importances = []
    feature_names = set()
    
    for model_info in loaded_top_models:
        calibrated_importances.append(dict(model_info['feature_importances_calibrated']))
        non_calibrated_importances.append(dict(model_info['feature_importances_non_calibrated']))
        feature_names.update([name for name, _ in model_info['feature_importances_calibrated']])
        feature_names.update([name for name, _ in model_info['feature_importances_non_calibrated']])
    
    # Create a DataFrame for easier plotting
    feature_importance_data = []
    for name in feature_names:
        for imp in calibrated_importances:
            if name in imp:
                feature_importance_data.append({'Feature': name, 'Importance': imp[name], 'Type': f'{model_name_prefix}_Calibrated'})
        for imp in non_calibrated_importances:
            if name in imp:
                feature_importance_data.append({'Feature': name, 'Importance': imp[name], 'Type': f'{model_name_prefix}_Non-Calibrated'})
    
    df_importances = pd.DataFrame(feature_importance_data)
    # # Check if file exists and append data or create a new file
    # if os.path.exists(filename):
    #     existing_df = pd.read_csv(filename)
    #     df_importances = pd.concat([existing_df, df_importances], ignore_index=True)
    df_importances.to_csv(filename, index=False)
    print(f"Data saved to - {filename}")

    return df_importances

# Function to plot individual calibration curves
def plot_individual_calibration_curves(y_true, predicted_proba_list, n_bins, model_label, calibrated=True):
    plt.figure(figsize=[12, 8])
    sns.set(style="whitegrid")
    
    for idx, prob in enumerate(predicted_proba_list):
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, prob, n_bins=n_bins)
        label = f"{model_label} Model {idx+1} ({'Calibrated' if calibrated else 'Non-Calibrated'})"
        sns.lineplot(x=mean_predicted_value, y=fraction_of_positives, marker="o", label=label)
    
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability", fontsize=14)
    plt.ylabel("Fraction of positives", fontsize=14)
    plt.title(f"Calibration curves for {model_label} Models ({'Calibrated' if calibrated else 'Non-Calibrated'})", fontsize=16)
    
    # Customize legend
    plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()


def create_ensemble_model(top_models, calibration=True):
    """
    Create an ensemble model from the top 10 models using VotingClassifier.

    Parameters:
    top_models (list): List of dictionaries containing the top models information.

    Returns:
    VotingClassifier: An ensemble model created from the top 10 models.
    """
    # Create a list of tuples with model names and the actual models
    if calibration:
        estimators = [(f'model_{i}', model_info['model_calibrated']) for i, model_info in enumerate(top_models)]
    else:
        estimators = [(f'model_{i}', model_info['model_non_calibrated']) for i, model_info in enumerate(top_models)]

    # Create a voting classifier
    ensemble_model = VotingClassifier(estimators=estimators, voting='soft')

    return ensemble_model


def evaluate_models(models, model_names, X_val, y_val, n_bootstrap=1000):
    """
    Evaluate a list of models and return the predicted probabilities and evaluation metrics, 
    including standard deviation and 95% CI for the metrics.

    Parameters:
    models (list): List of trained ensemble models.
    model_names (list): List of model names corresponding to the models.
    X_val (pd.DataFrame): Validation feature set.
    y_val (pd.Series): True labels for the validation set.
    n_bootstrap (int): Number of bootstrap resamples.

    Returns:
    pd.DataFrame: DataFrame containing predicted probabilities.
    pd.DataFrame: DataFrame containing evaluation metrics with std dev and 95% CI.
    """
    probabilities = {}
    metrics = []

    for model, name in zip(models, model_names):
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        # Store predicted probabilities
        probabilities[f'{name}_proba'] = y_proba

        # Bootstrap resampling to calculate metrics with std dev and 95% CI
        roc_aucs = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for i in range(n_bootstrap):
            X_resample, y_resample = resample(X_val, y_val)
            y_proba_resample = model.predict_proba(X_resample)[:, 1]
            y_pred_resample = model.predict(X_resample)

            fpr, tpr, _ = roc_curve(y_resample, y_proba_resample)
            roc_aucs.append(auc(fpr, tpr))
            accuracies.append(accuracy_score(y_resample, y_pred_resample))
            precisions.append(precision_score(y_resample, y_pred_resample))
            recalls.append(recall_score(y_resample, y_pred_resample))
            f1_scores.append(f1_score(y_resample, y_pred_resample))
        print(f"Bootstrapong Evaluation done for -- {name}")

        # Compute mean, std dev, and 95% CI for each metric
        metrics.append({
            'model': name,
            'roc_auc_mean': np.mean(roc_aucs),
            'roc_auc_std': np.std(roc_aucs),
            'roc_auc_ci_lower': np.percentile(roc_aucs, 2.5),
            'roc_auc_ci_upper': np.percentile(roc_aucs, 97.5),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_ci_lower': np.percentile(accuracies, 2.5),
            'accuracy_ci_upper': np.percentile(accuracies, 97.5),
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'precision_ci_lower': np.percentile(precisions, 2.5),
            'precision_ci_upper': np.percentile(precisions, 97.5),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'recall_ci_lower': np.percentile(recalls, 2.5),
            'recall_ci_upper': np.percentile(recalls, 97.5),
            'f1_score_mean': np.mean(f1_scores),
            'f1_score_std': np.std(f1_scores),
            'f1_score_ci_lower': np.percentile(f1_scores, 2.5),
            'f1_score_ci_upper': np.percentile(f1_scores, 97.5),
        })

    probabilities_df = pd.DataFrame(probabilities)
    metrics_df = pd.DataFrame(metrics)

    return probabilities_df, metrics_df