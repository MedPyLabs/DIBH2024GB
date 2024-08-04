import os
import pandas as pd

def save_results_to_csv(calibrated_summary, non_calibrated_summary, model_name, filename='final_results.csv'):
    """
    Save the summary statistics of calibrated and non-calibrated models to a CSV file.
    
    Parameters:
    calibrated_summary (pd.DataFrame): Summary statistics for calibrated models.
    non_calibrated_summary (pd.DataFrame): Summary statistics for non-calibrated models.
    model_name (str): The name of the model to be prepended to each entry in the "Models" column.
    filename (str): The name of the CSV file. Default is 'final_results.csv'.
    """
    # Combine calibrated and non-calibrated summaries into one DataFrame
    combined_summary = pd.DataFrame({
        'Models': [f'{model_name}_Calibrated', f'{model_name}_Non_Calibrated'],
        'Accuracy': [calibrated_summary.loc['mean', 'accuracy'], non_calibrated_summary.loc['mean', 'accuracy']],
        'Acc_Std': [calibrated_summary.loc['std', 'accuracy'], non_calibrated_summary.loc['std', 'accuracy']],
        'Acc_95%CI': [
            f"{calibrated_summary.loc['95% CI lower', 'accuracy']} - {calibrated_summary.loc['95% CI upper', 'accuracy']}",
            f"{non_calibrated_summary.loc['95% CI lower', 'accuracy']} - {non_calibrated_summary.loc['95% CI upper', 'accuracy']}"
        ],
        'Precision': [calibrated_summary.loc['mean', 'precision'], non_calibrated_summary.loc['mean', 'precision']],
        'Precision_Std': [calibrated_summary.loc['std', 'precision'], non_calibrated_summary.loc['std', 'precision']],
        'Precision_95%CI': [
            f"{calibrated_summary.loc['95% CI lower', 'precision']} - {calibrated_summary.loc['95% CI upper', 'precision']}",
            f"{non_calibrated_summary.loc['95% CI lower', 'precision']} - {non_calibrated_summary.loc['95% CI upper', 'precision']}"
        ],
        'Recall': [calibrated_summary.loc['mean', 'recall'], non_calibrated_summary.loc['mean', 'recall']],
        'Recall_Std': [calibrated_summary.loc['std', 'recall'], non_calibrated_summary.loc['std', 'recall']],
        'Recall_95%CI': [
            f"{calibrated_summary.loc['95% CI lower', 'recall']} - {calibrated_summary.loc['95% CI upper', 'recall']}",
            f"{non_calibrated_summary.loc['95% CI lower', 'recall']} - {non_calibrated_summary.loc['95% CI upper', 'recall']}"
        ],
        'F1Score': [calibrated_summary.loc['mean', 'f1_score'], non_calibrated_summary.loc['mean', 'f1_score']],
        'F1Score_Std': [calibrated_summary.loc['std', 'f1_score'], non_calibrated_summary.loc['std', 'f1_score']],
        'F1Score_95%CI': [
            f"{calibrated_summary.loc['95% CI lower', 'f1_score']} - {calibrated_summary.loc['95% CI upper', 'f1_score']}",
            f"{non_calibrated_summary.loc['95% CI lower', 'f1_score']} - {non_calibrated_summary.loc['95% CI upper', 'f1_score']}"
        ],
        'AUC': [calibrated_summary.loc['mean', 'roc_auc'], non_calibrated_summary.loc['mean', 'roc_auc']],
        'AUC_std': [calibrated_summary.loc['std', 'roc_auc'], non_calibrated_summary.loc['std', 'roc_auc']],
        'AUC_95%CI': [
            f"{calibrated_summary.loc['95% CI lower', 'roc_auc']} - {calibrated_summary.loc['95% CI upper', 'roc_auc']}",
            f"{non_calibrated_summary.loc['95% CI lower', 'roc_auc']} - {non_calibrated_summary.loc['95% CI upper', 'roc_auc']}"
        ],
    })

    # Check if the file exists
    if os.path.exists(filename):
        # Append to the existing file
        combined_summary.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Create a new file
        combined_summary.to_csv(filename, index=False)

    return f"Data saved to {filename}"

def get_results(results):
    # Convert results to a DataFrame
    df_results = pd.DataFrame(results)

    # Separate metrics for calibrated and non-calibrated models
    calibrated_metrics = df_results[['accuracy_calibrated', 'precision_calibrated', 'recall_calibrated', 'f1_score_calibrated', 'roc_auc_calibrated']]
    non_calibrated_metrics = df_results[['accuracy_non_calibrated', 'precision_non_calibrated', 'recall_non_calibrated', 'f1_score_non_calibrated', 'roc_auc_non_calibrated']]

    # Calculate summary statistics for calibrated metrics
    calibrated_summary = calibrated_metrics.agg(['mean', 'std', 'min', 'max'])
    calibrated_summary.loc['95% CI lower'] = calibrated_summary.loc['mean'] - 1.96 * calibrated_summary.loc['std']
    calibrated_summary.loc['95% CI upper'] = calibrated_summary.loc['mean'] + 1.96 * calibrated_summary.loc['std']

    # Calculate summary statistics for non-calibrated metrics
    non_calibrated_summary = non_calibrated_metrics.agg(['mean', 'std', 'min', 'max'])
    non_calibrated_summary.loc['95% CI lower'] = non_calibrated_summary.loc['mean'] - 1.96 * non_calibrated_summary.loc['std']
    non_calibrated_summary.loc['95% CI upper'] = non_calibrated_summary.loc['mean'] + 1.96 * non_calibrated_summary.loc['std']

    # Rename columns for clarity
    calibrated_summary.columns = [col.replace('_calibrated', '') for col in calibrated_summary.columns]
    non_calibrated_summary.columns = [col.replace('_non_calibrated', '') for col in non_calibrated_summary.columns]

    # Print the summaries
    print("Calibrated Model Metrics Summary")
    print(calibrated_summary)

    print("\nNon-Calibrated Model Metrics Summary")
    print(non_calibrated_summary)

    return calibrated_summary, non_calibrated_summary