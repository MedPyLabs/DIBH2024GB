
# Predicting DIBH Eligibility: A Machine Learning Approach

## Rationale

Deep Inspiration Breath Hold (DIBH) is a technique used in radiation therapy for left-sided breast cancer patients to minimize radiation exposure to the heart. The traditional method of assessing DIBH eligibility involves a three-day assessment process, which can be resource-intensive and burdensome on already constrained healthcare resources.

By predicting DIBH eligibility on the day of assessment, we can potentially save resources and reduce the workload on healthcare facilities. This project aims to develop and validate a machine learning model that can predict DIBH eligibility based on clinical data collected on the first day of assessment.

## Approach

This repository contains the code and data used to develop a predictive model for DIBH eligibility. The project involves several key steps:

	1.	Data Preprocessing: Cleaning and preparing the data for modeling.
	2.	Hyperparameter Optimization: Using Optuna to find the best hyperparameters for the model.
	3.	Bootstrap Sampling: Generating multiple resampled datasets to train and evaluate models.
	4.	Ensemble Modeling: Combining multiple models to create a robust predictive ensemble.
	5.	Internal Validation: Evaluating the model on an independent validation dataset.
	6.	Results: Summarizing the performance of the model.

## Important Libraries

	•	Pandas: For data manipulation and analysis.
	•	NumPy: For numerical computations.
	•	SciPy: For scientific and technical computing.
	•	Statsmodels: For statistical models.
	•	Scikit-learn: For machine learning and data preprocessing.
	•	Optuna: For hyperparameter optimization.
	•	Joblib: For saving and loading models.
	•	Matplotlib: For plotting and visualization.
	•	Seaborn: For statistical data visualization.

## Data Preprocessing

The data preparation involves loading the clinical data and filtering it to include only the records from the first day of assessment. The dataset is then split into training and validation sets. The features are divided into categorical and continuous variables, and preprocessing pipelines are created for each type:

	•	Numerical Pipeline: Imputes missing values with the mean and scales the features.
	•	Categorical Pipeline: Imputes missing values with the most frequent value and encodes the features using one-hot encoding.

These pipelines are combined using a ColumnTransformer to preprocess the data before modeling.

## Hyperparameter Optimization

Hyperparameter optimization is performed using Optuna, a hyperparameter optimization framework. Optuna is used to tune the hyperparameters of a Gradient Boosting Classifier. The objective function is defined to maximize the ROC AUC score through cross-validation. The best hyperparameters are determined by running multiple trials.

## Bootstrap Sampling

Bootstrap sampling is used to generate multiple resampled datasets. This technique helps in understanding the variability of the model and ensures that the model is robust. For each bootstrap sample, the data is split into training and validation sets, and the model is trained and evaluated.

## Ensemble Modeling

Ensemble modeling involves combining multiple models to create a robust predictive model. In this project, we use the top models obtained from the bootstrap samples to create an ensemble classifier. The ensemble classifier predicts DIBH eligibility by averaging the probabilities predicted by the individual models.

## Internal Validation

Internal validation is performed on an independent validation dataset. The ensemble classifier is evaluated on this dataset to ensure its performance is reliable. Metrics such as accuracy, ROC AUC, precision, recall, and F1 score are calculated to evaluate the model’s performance.

## Results

The results of the model evaluation are summarized to provide an overview of the model’s performance. The key metrics are aggregated to provide mean, standard deviation, and confidence intervals. The confusion matrix and classification report are also generated to provide a detailed understanding of the model’s performance.
