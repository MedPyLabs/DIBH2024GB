from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, 
                             classification_report, brier_score_loss, 
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
import os
import numpy as np
import pandas as pd


def new_bootstrap_model_evaluation(X_t, y_t, X_v, y_v, preprocessor, base_model, best_params=None, n_iterations=1000):
    # Initialize lists to store results
    top_models = []
    results = []

    # Initialize lists to store predicted probabilities
    predicted_proba_calibrated = []
    predicted_proba_non_calibrated = []

    # Initialize lists for top 10 model predicted probabilities
    top_predicted_proba_calibrated = []
    top_predicted_proba_non_calibrated = []

    for i in range(n_iterations):
        X_resampled, y_resampled = resample(X_t, y_t)

        # Create base model instance
        if best_params:
            if base_model == SVC:
                base_model_instance = base_model(probability=True, **best_params, random_state=42)
            else:
                try:
                    base_model_instance = base_model(**best_params, random_state=42)
                except TypeError:
                    base_model_instance = base_model(**best_params)
        else:
            if base_model == SVC:
                base_model_instance = base_model(probability=True, random_state=42)
            else:
                try:
                    base_model_instance = base_model(random_state=42)
                except TypeError:
                    base_model_instance = base_model()

        # Non-calibrated model
        pipeline_non_calibrated = make_pipeline(preprocessor, base_model_instance)
        pipeline_non_calibrated.fit(X_resampled, y_resampled)
        y_proba_non_calibrated = pipeline_non_calibrated.predict_proba(X_v)[:, 1]
        predicted_proba_non_calibrated.append(y_proba_non_calibrated)

        if best_params:
            if base_model == SVC:
                base_model_instance = base_model(probability=True, **best_params, random_state=42)
            else:
                try:
                    base_model_instance = base_model(**best_params, random_state=42)
                except TypeError:
                    base_model_instance = base_model(**best_params)
        else:
            if base_model == SVC:
                base_model_instance = base_model(probability=True, random_state=42)
            else:
                try:
                    base_model_instance = base_model(random_state=42)
                except TypeError:
                    base_model_instance = base_model()

        # Calibrated model
        pipeline_calibrated = make_pipeline(preprocessor, base_model_instance)
        pipeline_calibrated.fit(X_resampled, y_resampled)
        model = CalibratedClassifierCV(pipeline_calibrated, method='isotonic', cv='prefit')
        model.fit(X_v, y_v)
        y_proba_calibrated = model.predict_proba(X_v)[:, 1]
        predicted_proba_calibrated.append(y_proba_calibrated)

        # Verify that the models are fitted
        base_model_name = base_model.__name__.lower()
        try:
            check_is_fitted(pipeline_non_calibrated.named_steps[base_model_name])
        except NotFittedError:
            print(f"{base_model.__name__} in non-calibrated pipeline is not fitted. Skipping this model.")
            continue

        try:
            check_is_fitted(model)
        except NotFittedError:
            print(f"CalibratedClassifierCV is not fitted. Skipping this model.")
            continue

        # ROC curve and other metrics for calibrated model
        fpr, tpr, thresholds = roc_curve(y_v, y_proba_calibrated)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_optimal = (y_proba_calibrated >= optimal_threshold).astype(int)

        # ROC curve and other metrics for non-calibrated model
        fpr_non_calibrated, tpr_non_calibrated, thresholds_non_calibrated = roc_curve(y_v, y_proba_non_calibrated)
        roc_auc_non_calibrated = auc(fpr_non_calibrated, tpr_non_calibrated)
        optimal_idx_non_calibrated = np.argmax(tpr_non_calibrated - fpr_non_calibrated)
        optimal_threshold_non_calibrated = thresholds_non_calibrated[optimal_idx_non_calibrated]
        y_pred_optimal_non_calibrated = (y_proba_non_calibrated >= optimal_threshold_non_calibrated).astype(int)

        # Feature importance for non-calibrated model
        if hasattr(pipeline_non_calibrated.named_steps[base_model_name], 'feature_importances_'):
            feature_importances_non_calibrated = pipeline_non_calibrated.named_steps[base_model_name].feature_importances_
        else:
            X_v_transformed = preprocessor.transform(X_v)
            results_non_calibrated = permutation_importance(pipeline_non_calibrated.named_steps[base_model_name], X_v_transformed, y_v, n_repeats=30, random_state=42, n_jobs=-1)
            feature_importances_non_calibrated = results_non_calibrated.importances_mean

        preprocessor = pipeline_non_calibrated.named_steps['columntransformer']
        feature_names = preprocessor.get_feature_names_out()

        # Aggregate importances for one-hot encoded features for non-calibrated model
        aggregated_importances_non_calibrated = defaultdict(float)
        for feature_name, importance in zip(feature_names, feature_importances_non_calibrated):
            original_feature_name = feature_name.split('__')[1].split('_')[0] if 'cat__' in feature_name else feature_name.split('__')[1]
            aggregated_importances_non_calibrated[original_feature_name] += importance

        # Convert aggregated importances to a sorted list of tuples for non-calibrated model
        sorted_importances_non_calibrated = sorted(aggregated_importances_non_calibrated.items(), key=lambda x: x[1], reverse=True)

        # Feature importance for calibrated model
        if hasattr(model.estimator.named_steps[base_model_name], 'feature_importances_'):
            feature_importances_calibrated = model.estimator.named_steps[base_model_name].feature_importances_
        else:
            X_v_transformed = preprocessor.transform(X_v)
            results_calibrated = permutation_importance(model.estimator.named_steps[base_model_name], X_v_transformed, y_v, n_repeats=30, random_state=42, n_jobs=-1)
            feature_importances_calibrated = results_calibrated.importances_mean

        # Aggregate importances for one-hot encoded features for calibrated model
        aggregated_importances_calibrated = defaultdict(float)
        for feature_name, importance in zip(feature_names, feature_importances_calibrated):
            original_feature_name = feature_name.split('__')[1].split('_')[0] if 'cat__' in feature_name else feature_name.split('__')[1]
            aggregated_importances_calibrated[original_feature_name] += importance

        # Convert aggregated importances to a sorted list of tuples for calibrated model
        sorted_importances_calibrated = sorted(aggregated_importances_calibrated.items(), key=lambda x: x[1], reverse=True)

        model_info = {
            'model_calibrated': model,
            'fpr_calibrated': fpr,
            'tpr_calibrated': tpr,
            'thresholds_calibrated': thresholds,
            'roc_auc_calibrated': roc_auc,
            'optimal_threshold_calibrated': optimal_threshold,
            'accuracy_calibrated': accuracy_score(y_v, y_pred_optimal),
            'precision_calibrated': precision_score(y_v, y_pred_optimal),
            'recall_calibrated': recall_score(y_v, y_pred_optimal),
            'f1_score_calibrated': f1_score(y_v, y_pred_optimal),
            'confusion_matrix_calibrated': confusion_matrix(y_v, y_pred_optimal),
            'classification_report_calibrated': classification_report(y_v, y_pred_optimal),
            'feature_importances_calibrated': sorted_importances_calibrated,
            'model_non_calibrated': pipeline_non_calibrated,
            'fpr_non_calibrated': fpr_non_calibrated,
            'tpr_non_calibrated': tpr_non_calibrated,
            'thresholds_non_calibrated': thresholds_non_calibrated,
            'roc_auc_non_calibrated': roc_auc_non_calibrated,
            'optimal_threshold_non_calibrated': optimal_threshold_non_calibrated,
            'accuracy_non_calibrated': accuracy_score(y_v, y_pred_optimal_non_calibrated),
            'precision_non_calibrated': precision_score(y_v, y_pred_optimal_non_calibrated),
            'recall_non_calibrated': recall_score(y_v, y_pred_optimal_non_calibrated),
            'f1_score_non_calibrated': f1_score(y_v, y_pred_optimal_non_calibrated),
            'confusion_matrix_non_calibrated': confusion_matrix(y_v, y_pred_optimal_non_calibrated),
            'classification_report_non_calibrated': classification_report(y_v, y_pred_optimal_non_calibrated),
            'feature_importances_non_calibrated': sorted_importances_non_calibrated
        }

        # Add the non-model info to results
        model_info_without_model = {key: value for key, value in model_info.items() if not key.startswith('model')}
        results.append(model_info_without_model)

        # Keep top 10 models based on calibrated ROC AUC
        if len(top_models) < 10:
            top_models.append(model_info)
            top_predicted_proba_calibrated.append(y_proba_calibrated)
            top_predicted_proba_non_calibrated.append(y_proba_non_calibrated)
        else:
            min_index = min(range(len(top_models)), key=lambda x: (top_models[x]["roc_auc_calibrated"], top_models[x]["recall_calibrated"]))
            if roc_auc > top_models[min_index]["roc_auc_calibrated"]:
                top_models[min_index] = model_info
                top_predicted_proba_calibrated[min_index] = y_proba_calibrated
                top_predicted_proba_non_calibrated[min_index] = y_proba_non_calibrated

        if (i + 1) % 25 == 0:
            print(f"Bootstrap sample no. {i + 1} ------ Finished")

    calibration_data = {
    'predicted_proba_calibrated': predicted_proba_calibrated,
    'predicted_proba_non_calibrated': predicted_proba_non_calibrated,
    'top_predicted_proba_calibrated': top_predicted_proba_calibrated,
    'top_predicted_proba_non_calibrated': top_predicted_proba_non_calibrated
    }

    return top_models, results, calibration_data



def get_calibrated_model(X_train, y_train, X_val, y_val, preprocessor, base_model, best_params=None, model_name_prefix="model", csv_filename="model_metrics.csv"):
    # Prepare the base model with or without parameters
    if best_params:
        if base_model == SVC:
            base_model_instance = base_model(probability=True, **best_params, random_state=42)
        else:
            try:
                base_model_instance = base_model(**best_params, random_state=42)
            except TypeError:
                base_model_instance = base_model(**best_params)
    else:
        if base_model == SVC:
            base_model_instance = base_model(probability=True, random_state=42)
        else:
            try:
                base_model_instance = base_model(random_state=42)
            except TypeError:
                base_model_instance = base_model()
    
    # Non-calibrated model pipeline
    pipeline_non_calibrated = make_pipeline(preprocessor, base_model_instance)
    pipeline_non_calibrated.fit(X_train, y_train)
    y_proba_non_calibrated = pipeline_non_calibrated.predict_proba(X_val)[:, 1]
    
    # Calibrated model pipeline
    model = CalibratedClassifierCV(pipeline_non_calibrated, method='isotonic', cv='prefit')
    model.fit(X_val, y_val)
    y_proba_calibrated = model.predict_proba(X_val)[:, 1]

    # Metrics for calibrated model
    brier_score_calibrated = brier_score_loss(y_val, y_proba_calibrated)
    fpr, tpr, _ = roc_curve(y_val, y_proba_calibrated)
    auc_calibrated = auc(fpr, tpr)
    accuracy_calibrated = accuracy_score(y_val, (y_proba_calibrated >= 0.5).astype(int))
    precision_calibrated = precision_score(y_val, (y_proba_calibrated >= 0.5).astype(int))
    recall_calibrated = recall_score(y_val, (y_proba_calibrated >= 0.5).astype(int))
    f1_calibrated = f1_score(y_val, (y_proba_calibrated >= 0.5).astype(int))

    # Metrics for non-calibrated model
    brier_score_non_calibrated = brier_score_loss(y_val, y_proba_non_calibrated)
    fpr_non_calibrated, tpr_non_calibrated, _ = roc_curve(y_val, y_proba_non_calibrated)
    auc_non_calibrated = auc(fpr_non_calibrated, tpr_non_calibrated)
    accuracy_non_calibrated = accuracy_score(y_val, (y_proba_non_calibrated >= 0.5).astype(int))
    precision_non_calibrated = precision_score(y_val, (y_proba_non_calibrated >= 0.5).astype(int))
    recall_non_calibrated = recall_score(y_val, (y_proba_non_calibrated >= 0.5).astype(int))
    f1_non_calibrated = f1_score(y_val, (y_proba_non_calibrated >= 0.5).astype(int))

    # Store metrics in dictionaries
    metrics_calibrated = {
        'Model': f'{model_name_prefix}_calibrated',
        'Brier Score': brier_score_calibrated,
        'AUC': auc_calibrated,
        'Accuracy': accuracy_calibrated,
        'Precision': precision_calibrated,
        'Recall': recall_calibrated,
        'F1 Score': f1_calibrated
    }

    metrics_non_calibrated = {
        'Model': f'{model_name_prefix}_non_calibrated',
        'Brier Score': brier_score_non_calibrated,
        'AUC': auc_non_calibrated,
        'Accuracy': accuracy_non_calibrated,
        'Precision': precision_non_calibrated,
        'Recall': recall_non_calibrated,
        'F1 Score': f1_non_calibrated
    }

    # Save metrics to CSV file
    save_metrics_to_csv(metrics_calibrated, metrics_non_calibrated, csv_filename)

    return model, metrics_calibrated, metrics_non_calibrated

def save_metrics_to_csv(metrics_calibrated, metrics_non_calibrated, filename):
    df_calibrated = pd.DataFrame([metrics_calibrated])
    df_non_calibrated = pd.DataFrame([metrics_non_calibrated])
    
    if os.path.exists(filename):
        df_existing = pd.read_csv(filename)
        df = pd.concat([df_existing, df_calibrated, df_non_calibrated], ignore_index=True)
    else:
        df = pd.concat([df_calibrated, df_non_calibrated], ignore_index=True)
    
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")