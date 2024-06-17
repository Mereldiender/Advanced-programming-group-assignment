import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

# Function to read data from a file
def read_data(input_file):
    data = pd.read_pickle(input_file)
    return data

# Function to find hyperparameters for RandomForestClassifier
def find_hyperparameters(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': list(range(1, 21)),
        'max_depth': [None] + list(np.linspace(3, 100, num=100).astype(int)),
        'max_features': ['sqrt', 'log2', None] + list(np.arange(0.1, 1.1, 0.1)),
        'max_leaf_nodes': [None] + list(np.linspace(10, 100, num=100).astype(int)),
        'min_samples_split': [2, 5, 10, 15],
        'bootstrap': [True, False]
    }
    rfc = RandomForestClassifier()
    rs = RandomizedSearchCV(rfc, param_grid, n_jobs=-1, scoring='balanced_accuracy', cv=3, n_iter=10, verbose=1, random_state=42)
    rs.fit(X_train, y_train)
    best_model = rs.best_estimator_
    best_params = rs.best_params_
    prediction = best_model.predict(X_test)
    bal_acc_score = balanced_accuracy_score(y_test, prediction)
    print(f'Model balanced accuracy score after hyperparameter tuning: {bal_acc_score:.4f}')
    return best_model, best_params, bal_acc_score

# Function to find models using cross-validation
def find_models(inhibition, train_data_file):
    data = read_data(train_data_file)
    X = data.drop(columns=['PKM2_inhibition', 'SMILES', 'ERK2_inhibition'], axis=1)
    y = data[inhibition]
    kf = KFold(n_splits=5, shuffle=True)
    models_CV = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        best_model, best_params, bal_acc_score = find_hyperparameters(X_train, X_test, y_train, y_test)
        models_CV.append(best_model)
    return models_CV

# Function to train SVM model
def models_svm(model, inhibition, train_data_file):
    data = read_data(train_data_file)
    X = data.drop(columns=['PKM2_inhibition', 'SMILES', 'ERK2_inhibition'], axis=1)
    y = data[inhibition]
    model.fit(X, y)
    return model

# Function to train XGBoost model
def models_xgboost(model, inhibition, train_data_file):
    data = read_data(train_data_file)
    X = data.drop(columns=['PKM2_inhibition', 'SMILES', 'ERK2_inhibition'], axis=1)
    y = data[inhibition]
    class_weights = compute_sample_weight(class_weight='balanced', y=y)
    model.fit(X, y, sample_weight=class_weights)
    return model

# Function to average predictions from RandomForest models
def average_predictionRF(models_CV, X_test):
    cv_predictions = []
    for model in models_CV:
        y_pred = model.predict(X_test)
        cv_predictions.append(y_pred)
    rf_cv_preds_stacked = np.column_stack(cv_predictions)
    rf_avg_preds = np.mean(rf_cv_preds_stacked, axis=1)
    rf_final_preds = (rf_avg_preds >= 0.4).astype(int)
    return rf_final_preds

# Function to predict with models
def predict_model(models_CV, X_test):
    y_preds = []
    for model in models_CV:
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
    return y_preds

# Function to apply voting mechanism on predictions
def voting_mechanism(predictions):
    preds_list = list(predictions.values())
    stacked_preds = np.column_stack(preds_list)
    final_preds = np.any(stacked_preds == 1, axis=1).astype(int)
    return final_preds

# Load train data files
train_data_file = 'Data/train_descriptors_balanced_pc_80.pkl'
train_data_file_bin = 'Data/training_fingerprints_balanced.pkl'

# Train RandomForest models
models_CV_PKM2 = find_models('PKM2_inhibition', train_data_file)
models_CV_ERK2 = find_models('ERK2_inhibition', train_data_file)
models_CV_PKM2_bin = find_models('PKM2_inhibition', train_data_file_bin)
models_CV_ERK2_bin = find_models('ERK2_inhibition', train_data_file_bin)

# Train SVM models
svm_PKM2 = SVC(class_weight='balanced', probability=True, gamma=0.5, C=1)
models_SVM_PKM2 = models_svm(svm_PKM2, 'PKM2_inhibition', train_data_file)
svm_ERK2 = SVC(class_weight='balanced', probability=True, gamma=1, C=1)
models_SVM_ERK2 = models_svm(svm_ERK2, 'ERK2_inhibition', train_data_file)

# Train XGBoost models
xgb_PKM2 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
models_xgb_PKM2 = models_xgboost(xgb_PKM2, 'PKM2_inhibition', train_data_file)
xgb_ERK2 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
models_xgb_ERK2 = models_xgboost(xgb_ERK2, 'ERK2_inhibition', train_data_file)

# Read untested data files
untested_fingerprints = 'Data/untested_fingerprints_2.pkl'
untested_pca = 'Data/untested_pca_2.pkl'
test_data_file_bin = untested_fingerprints
test_data_file = untested_pca

# Read test data
test_data = read_data(test_data_file)
X_test = test_data.drop(columns=['PKM2_inhibition', 'SMILES', 'ERK2_inhibition'], axis=1)

test_data_bin = read_data(test_data_file_bin)
X_test_bin = test_data_bin.drop(columns=['PKM2_inhibition', 'SMILES', 'ERK2_inhibition'], axis=1)

# Predict with RandomForest models and store predictions in a dictionary
predictions_PKM2 = {}
predictions_PKM2['RandomForest'] = average_predictionRF(models_CV_PKM2, X_test)
predictions_PKM2['RandomForest_fingerprints'] = average_predictionRF(models_CV_PKM2_bin, X_test_bin)

predictions_ERK2 = {}
predictions_ERK2['RandomForest'] = average_predictionRF(models_CV_ERK2, X_test)
predictions_ERK2['RandomForest_fingerprints'] = average_predictionRF(models_CV_ERK2_bin, X_test_bin)

# Predict with SVM models and store predictions in the dictionary
predictions_PKM2['SVM'] = models_SVM_PKM2.predict(X_test)
predictions_ERK2['SVM'] = models_SVM_ERK2.predict(X_test)

# Predict with XGBoost models and store predictions in the dictionary
predictions_PKM2['XGBoost'] = models_xgb_PKM2.predict(X_test)
predictions_ERK2['XGBoost'] = models_xgb_ERK2.predict(X_test)

# Apply voting mechanism to get final predictions
final_preds_PKM2 = voting_mechanism(predictions_PKM2)
final_preds_ERK2 = voting_mechanism(predictions_ERK2)

# Load the untested molecules CSV file
untested_molecules_file = 'Data/untested_molecules-3.csv'
untested_molecules = pd.read_csv(untested_molecules_file)

# Update the NaN values in PKM2_inhibition and ERK2_inhibition columns with the predictions
untested_molecules.loc[untested_molecules['PKM2_inhibition'].isna(), 'PKM2_inhibition'] = final_preds_PKM2
untested_molecules.loc[untested_molecules['ERK2_inhibition'].isna(), 'ERK2_inhibition'] = final_preds_ERK2

# Convert the values to integers
untested_molecules['PKM2_inhibition'] = untested_molecules['PKM2_inhibition'].astype(int)
untested_molecules['ERK2_inhibition'] = untested_molecules['ERK2_inhibition'].astype(int)

# Save the updated CSV file
output_file = 'Data/untested_molecules-3_updated.csv'
untested_molecules.to_csv(output_file, index=False)

print(f'Updated file saved to {output_file}')

# Count and print the number of molecules with PKM2 inhibition and ERK2 inhibition
pkm2_inhibition_count = untested_molecules['PKM2_inhibition'].sum()
erk2_inhibition_count = untested_molecules['ERK2_inhibition'].sum()

print(f'Number of molecules with PKM2 inhibition: {pkm2_inhibition_count}')
print(f'Number of molecules with ERK2 inhibition: {erk2_inhibition_count}')

