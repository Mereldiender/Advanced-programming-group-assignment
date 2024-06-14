# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:02:27 2024

@author: 20192547
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, KFold,train_test_split
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay,confusion_matrix
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

## ---------------- Read file do dataframe
def read_data(input_file):
    """
    Reads the input file and returns a dataframe.

    Parameters
    ----------
    input_file : str
        Name of the input file in string format
    
    Returns
    -------
    data : dataframe
        Dataframe of the information in input file
    """
    data = pd.read_pickle(input_file)
    return data


## ---------------- Using random search to optimize random forest
def find_hyperparameters(X_train, X_test, y_train, y_test):
    """
    Returns the parameters for the random forest classifier that result in 
    the highest balanced accuracy score.

    Parameters
    ----------
    X_train : dataframe
        Dataframe containing the feature values used for training the model
    X_test : dataframe
        Dataframe containing the feature values used for testing the model
    y_train : dataframe
        Dataframe containing the output variables used to train the model
    y_test : dataframe
        Dataframe containing the output variables to be predicted

    Returns
    -------
    best_model : RandomForestClassifier
        Random forest classifier that results in the best balanced
        accuracy score
    best_params : dict
        Dictionary containing the values of the hyperparameters that result
        in the best balanced accuracy score
    bal_acc_score : float
        Balanced accuracy score of the value to be predicted
    """
    # Hyperparameter grid
    param_grid = {
        'n_estimators' : list(range(1,21)),                                     # Amount of trees to use
        'max_depth': [None] + list(np.linspace(
            3, 100, num=100).astype(int)
            ),                                                                  # Maximum number of levels in tree
        'max_features': ['sqrt', 'log2', None] + list(
            np.arange(0.1, 1.1, 0.1)
            ),                                                                  # Number of features to consider at every split
        'max_leaf_nodes': [None] + list(np.linspace(
            10, 100, num=100).astype(int)
            ),                                                                  # How many leaf nodes can be visited
        'min_samples_split': [2, 5, 10, 15],                                    # Minimum number of samples required to split a node
        'bootstrap': [True, False]                                              # Method of selecting samples for training each tree
    }

    # Estimator for use in random search
    rfc = RandomForestClassifier()

    # Create the random search model
    rs = RandomizedSearchCV(rfc, param_grid, n_jobs=-1, 
                            scoring='balanced_accuracy', cv=3, 
                            n_iter=10, verbose=1, random_state=None)

    # Fit 
    rs.fit(X_train, y_train)
    best_model = rs.best_estimator_
    best_params = rs.best_params_ 
    prediction = best_model.predict(X_test)
    
    # Calculate the balanced accuracy score for each target
    bal_acc_score = balanced_accuracy_score(y_test, prediction)
    print('Model balanced accuracy score after hyperparameter tuning: {0:0.4f}'.format(
        bal_acc_score)
        )
    return best_model, best_params, bal_acc_score


## ---------------- Find five models by use of cross validation
def find_models(inhibition, train_data_file):
    """
    Performs cross-validation on the train dataset,
    returns five different models

    Parameters
    ----------
    inhibition : str
        The name of the variable (kinase) to be predicted
    train_data_file : str
        The name of the file that contains the data
    
    Returns
    -------
    models_CV : list of dict
        List containing a dict per model created using the random
        forest classifier
    """
    data = read_data(train_data_file)
    # Features
    X = data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    
    # Values to be predicted
    y = data[inhibition]

    kf = KFold(n_splits=5, shuffle=True)

    models_CV = []
    
    i = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model, best_params, bal_acc_score = find_hyperparameters(X_train, X_test, y_train, y_test)
        models_CV.append(best_model)
        
        i += 1
        
    return models_CV

def models_svm(model,inhibition, train_data_file):
    data = read_data(train_data_file)
    # Features
    X = data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    # Values to be predicted
    y = data[inhibition]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    model.fit(X_train, y_train)   
    return model

def models_xgboost(model,inhibition, train_data_file):
    data = read_data(train_data_file)
    # Features
    X = data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    
    # Values to be predicted
    y = data[inhibition]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    # Compute class weights
    class_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train, y_train,sample_weight=class_weights)        
    return model

## ---------------- Confusion Matrix
# Print the Confusion Matrix and slice it into four pieces
def create_confusion_matrix(y_test, y_pred, rfc, title):
    cm = confusion_matrix(y_test, y_pred, labels=rfc.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
    disp.plot()
    plt.title(title)
    plt.show()

def average_predictionRF(models_CV):
    cv_predictions=[]
    for i, rfc_model in enumerate(models_CV, start=1):
        y_pred = rfc_model['y_pred']
        cv_predictions.append(y_pred)

    # Stack the predictions and average them
    rf_cv_preds_stacked = np.column_stack(cv_predictions)
    rf_avg_preds = np.mean(rf_cv_preds_stacked, axis=1)

    # Convert averaged predictions to binary (0 or 1) based on the 0.4 threshold
    rf_final_preds = (rf_avg_preds >= 0.4).astype(int)
    return rf_final_preds

def predict_model_training(models_CV, X_test,y_test): #used during training when you have the y_test
    Models_pred=[]
    for fitted_model in models_CV:
        y_pred = fitted_model.predict(X_test)
        bal_acc_score = balanced_accuracy_score(y_test, y_pred)
        Models_pred.append({'Model 1': fitted_model, 
                          'y_test': y_test, 'y_pred': y_pred, 'BAcc': bal_acc_score})
    return Models_pred

def predict_model(models_CV, X_test):
    Models_pred=[]
    for fitted_model in models_CV:
        y_pred = fitted_model.predict(X_test)
        Models_pred.append({'Model 1': fitted_model, 
                           'y_pred': y_pred})
    return Models_pred

## ---------------- Call the functions to create the models
# Insert paths to files where input_file contains all the testing and
# training data, train_data_file only contains the training data and
# test_data_file only contains the data used for testing.

train_data_file = 'C:\\Users\\20192547\\OneDrive - TU Eindhoven\\Documents\\JAAR 5\\Advanced programming and biomedical data analysis\\Group assignment\\train_descriptors_balanced_pc_80.pkl'
train_data_file_bin= 'C:\\Users\\20192547\\OneDrive - TU Eindhoven\\Documents\\JAAR 5\\Advanced programming and biomedical data analysis\\Group assignment\\training_fingerprints_balanced.pkl'
#test_data_file = 'C:\\Users\\20212435\\Documents\\GitHub\\Group assignment\\Advanced-programming-group-assignment\\test_descriptors_balanced.pkl'

#______________TRAIN descriptor models______________
# Non-binary descriptor models
#_______RANDOM FOREST______
# PKM2 Models
models_CV_PKM2 = find_models('PKM2_inhibition', train_data_file)
# ERK2 Models
models_CV_ERK2 = find_models('ERK2_inhibition', train_data_file)


#_______SVM_______
# PKM2 Models
svm_PKM2 = SVC(class_weight='balanced', probability=True, gamma=0.5, C=1)
models_SVM_PKM2 = models_svm(svm_PKM2,'PKM2_inhibition', train_data_file)
# ERK2 Models
svm_ERK2 = SVC(class_weight='balanced', probability=True, gamma=1, C=1)
models_SVM_ERK2= models_svm(svm_ERK2,'ERK2_inhibition', train_data_file)

#_______XGBOOST_______
# PKM2 Models
xgb_PKM2 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
models_xgb_PKM2=models_xgboost(xgb_PKM2,'PKM2_inhibition', train_data_file)
# ERK2 Models
xgb_ERK2 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
models_xgb_ERK2=models_xgboost(xgb_ERK2,'ERK2_inhibition', train_data_file)

# Binary descriptor models
#_______RANDOM FOREST______
# PKM2 Models
models_CV_PKM2_bin = find_models('PKM2_inhibition', train_data_file_bin)
# ERK2 Models
models_CV_ERK2_bin = find_models('ERK2_inhibition', train_data_file_bin)


#______________PREDICT NON-BINARY descriptor models______________
predictions_PKM2={}
predictions_ERK2={}
# Get test data
#for non-binary models
data = read_data(train_data_file)
X = data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
y = data['PKM2_inhibition']
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=10)
#for binary models
data = read_data(train_data_file_bin)
X = data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
y = data['PKM2_inhibition']
_, X_test_bin, _, _ = train_test_split(X, y, test_size=0.2, random_state=10)

#_______RANDOM FOREST Non-binary_______
# PKM2 Models
CV_pred_PKM2=predict_model(models_CV_PKM2, X_test)   
rf_pred_pkm2=average_predictionRF(CV_pred_PKM2)
predictions_PKM2['RandomForest']=rf_pred_pkm2
# ERK2 Models
CV_pred_ERK2=predict_model(models_CV_ERK2, X_test)   
rf_pred_ERK2=average_predictionRF(CV_pred_ERK2)
predictions_ERK2['RandomForest']=rf_pred_ERK2

#_______SVM_______
# PKM2 Models
#make list of models
models_SVM_PKM2_list=[models_SVM_PKM2]
SVM_pred_PKM2=predict_model(models_SVM_PKM2_list, X_test) 
predictions_PKM2['SVM']=SVM_pred_PKM2[0]['y_pred'] 
# ERK2 Models
models_SVM_ERK2_list=[models_SVM_ERK2]
SVM_pred_ERK2=predict_model(models_SVM_ERK2_list, X_test) 
predictions_ERK2['SVM']=SVM_pred_ERK2[0]['y_pred'] 

#_______XGBOOST_______
# PKM2 Models
models_xgb_PKM2_list=[models_xgb_PKM2]
xgb_pred_PKM2=predict_model(models_xgb_PKM2_list, X_test) 
predictions_PKM2['xgb']=xgb_pred_PKM2[0]['y_pred'] 
# ERK2 Models
models_xgb_ERK2_list=[models_xgb_ERK2]
xgb_pred_ERK2=predict_model(models_xgb_ERK2_list, X_test) 
predictions_ERK2['xgb']=xgb_pred_ERK2[0]['y_pred'] 

#_______RANDOM FOREST_______
# PKM2 Models
CV_pred_PKM2_bin=predict_model(models_CV_PKM2_bin, X_test_bin)   
rf_pred_pkm2_bin=average_predictionRF(CV_pred_PKM2_bin)
predictions_PKM2['RandomForest_fingerprints']=rf_pred_pkm2_bin
# ERK2 Models
CV_pred_ERK2_bin=predict_model(models_CV_ERK2_bin, X_test_bin)   
rf_pred_ERK2_bin=average_predictionRF(CV_pred_ERK2_bin)
predictions_ERK2['RandomForest_fingerprints']=rf_pred_ERK2_bin
#%%  
#______________VOTING____________
def voting_mechanism(predictions):
    # Convert dictionary values to a list of numpy arrays
    preds_list = list(predictions.values())
    # Stack predictions horizontally to get a 2D array
    stacked_preds = np.column_stack(preds_list)
    # Apply the voting rule: 1 if any model predicts 1, else 0
    final_preds = np.any(stacked_preds == 1, axis=1).astype(int)
    return final_preds

final_preds_PKM2=voting_mechanism(predictions_PKM2)
final_preds_ERK2=voting_mechanism(predictions_ERK2)

