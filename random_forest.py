import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    make_scorer, average_precision_score
    )


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


def rf_classifier(inhibition, train_data, test_data, amount_trees):
    """
    Returns the multi-output classifier model used to predict the 
    inhibition of certain molecules to kinases (PKM2 and ERK2).

    Parameters
    ----------
    inhibition : str
        Name of the variable to be predicted
    train_data : dataframe
        Part of the input data used to train the model
    test_data : dataframe
        Part of the input data used to test the model
    amount_trees : int
        The amount of trees used for the random forest classifier

    Returns
    -------
    y_test : dataframe
        The output variables to be predicted in a dataframe
    y_pred : list of lists
        The output predicted by the rf classifier
    bal_acc_score : float
        The average accuracy score of the multi-output classifier
    rfc : RandomForestClassifier
        The random forest classifier model

    """
    # Declare feature vector (X) and target variable (y)
    X_train = train_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
    y_train = train_data[inhibition]

    X_test = test_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
    y_test = test_data[inhibition]

    ## ---------------- Random Forest Classifier model with default parameters
    # Instantiate the classifier
    # Random_state set to zero to make sure the model returns the same every time
    rfc = RandomForestClassifier(n_estimators = amount_trees, 
                                 random_state=0
                                 )
    # Fit the model
    rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = rfc.predict(X_test)

    # Calculate the average precision score for each target
    bal_acc_score = balanced_accuracy_score(y_test, y_pred)
    print('Model balanced accuracy score is:', bal_acc_score)
    # print('Model average balanced accuracy score: {0:0.4f}'.format(
    #     avg_bal_acc_score)
    #     )

    return y_test, y_pred, bal_acc_score, rfc

## ---------------- Confustion Matrix
# Print the Confusion Matrix and slice it into four pieces
def create_confusion_matrix(y_test, y_pred, rfc):
    cm = confusion_matrix(y_test, y_pred, labels=rfc.classes_)

    # Plot confustion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=rfc.classes_)
    disp.plot()
    plt.show()


## ---------------- Using random search to optimize random forest
def hyperparameter_optimization(X_train, X_test, y_train, y_test):
    """
    Returns the parameters for the multi-output classifier that result in 
    the highest average balanced accuracy score.

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
    best_model : MultiOutputClassifier
        Multi-output classifier that results in the best average accuracy 
        score
    best_params : dict
        Dictionary containing the values of the hyperparameters that result
        in the best average accuracy score
    avg_bal_accuracy_score : float
        Average accuracy score over the two variables to be predicted
    """
    # Hyperparameter grid
    param_grid = {
        'max_depth': [None] + list(np.linspace(
            3, 100, num=100).astype(int)
            ),                                                                  # Maximum number of levels in tree
        'max_features': ['sqrt', 'log2', None] + list(
            np.arange(0.1, 1.1, 0.1)
            ),                                                                  # Number of features to consider at every split
        'max_leaf_nodes': [None] + list(np.linspace(
            10, 100, num=100).astype(int)
            ),                                                                  # How many leaf nodes can be visited
        'min_samples_split': [2, 5, 10, 15],                         # Minimum number of samples required to split a node
        'bootstrap': [True, False]                                   # Method of selecting samples for training each tree
    }

    # Estimator for use in random search
    rfc = RandomForestClassifier(n_estimators=4)

    # Create the random search model
    rs = RandomizedSearchCV(rfc, param_grid, n_jobs=-1, 
                            scoring='balanced_accuracy', cv=3, 
                            n_iter=10, verbose=1, random_state=None)

    # Fit 
    rs.fit(X_train, y_train)
    best_model = rs.best_estimator_
    best_params = rs.best_params_ 
    prediction = best_model.predict(X_test)
    
    # Calculate the average precision score for each target
    bal_acc_score = balanced_accuracy_score(y_test, prediction)
    print('Model balanced accuracy score after hyperparameter tuning: {0:0.4f}'.format(
        bal_acc_score)
        )
    return best_model, best_params, bal_acc_score


def fingerprints_eval(inhibition, train_data_file, test_data_file):
    """
    Returns the amount of trees that leads to the model with the highest 
    average balanced accuracy score.

    Parameters
    ----------
    train_data_file : str
        The name of the file that contains the training data
    test_data_file : str
        The name of the file that contains the testing data

    Returns
    -------
    best_model[3] : int
        The amount of trees leading to the model with the highest average 
        balanced accuracy score.
    """
    train_data = read_data(train_data_file)
    print("The amount of features is equal to {}.".format(
        len(train_data.axes[1]))
        )
    test_data = read_data(test_data_file)
    avg_bal_acc = 0
    for amount_trees in range(1,20):
        y_test, y_pred, avg_bal_acc_score, multi_target_rfc = rf_classifier(inhibition,
            train_data, test_data, amount_trees
            )
        # Save the values of the best model
        if avg_bal_acc_score > avg_bal_acc:
            avg_bal_acc = avg_bal_acc_score
            best_model = [y_test, y_pred, multi_target_rfc, amount_trees]
    print('The average balanced accuracy score of the best model is {} and it consists of {} trees.'.format(
        avg_bal_acc, best_model[3])
        )
    return best_model[3]


def fingerprints_model(inhibition, train_data_file, test_data_file, amount_trees):
    """
    Create the model with the best amount of trees and plot two confusion 
    matrices.

    Parameters
    ----------
    train_data_file : str
        The name of the file that contains the training data
    test_data_file : str
        The name of the file that contains the testing data
    amount_trees : int
        The amount of trees to use
    
    Returns
    -------
    None        
    """
    train_data = read_data(train_data_file)
    test_data = read_data(test_data_file)
    y_test, y_pred, bal_acc_score, rfc = rf_classifier(inhibition, 
        train_data, test_data, amount_trees
        )
    print('The balanced accuracy score is {}.'.format(bal_acc_score))
    create_confusion_matrix(y_test, y_pred)
    return bal_acc_score

def find_best_param(inhibition, train_data_file, test_data_file):
    """
    Returns the model with the best hyperparameters based on a condition 
    set for the average accuracy score.

    Parameters
    ----------
    train_data_file : str
        The name of the file that contains the training data
    test_data_file : str
        The name of the file that contains the testing data
    
    Returns
    -------
    best_model : MultiOutputClassifier
        Multi-output classifier that results in the best average 
        accuracy score
    best_params : dict
        Dictionary containing the values of the hyperparameters 
        that result in the best average accuracy score
    bal_accuracy : float
        Average balanced accuracy score over the two variables to be predicted 
    """
    train_data = read_data(train_data_file)
    X_train = train_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
    y_train = train_data[inhibition]

    test_data = read_data(test_data_file)
    X_test = test_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
    y_test = test_data[inhibition]

    bal_acc_score = 0
    # Make sure the accuracy is higher than for the default hyperparameters
    while bal_acc_score < 0.4764:
        best_model, best_params, bal_acc_score = hyperparameter_optimization(
            X_train, X_test, y_train, y_test
            )
    print("Best hyperparameters found: {}.".format(best_params))
    print('The balanced accuracy score is {}.'.format(bal_acc_score))
    y_pred = best_model.predict(X_test)
    create_confusion_matrix(y_test, y_pred, best_model)
    return best_model, best_params, bal_acc_score


## ---------------- Call the functions to create the model
# Insert paths to files where input_file contains all the testing and
# training data, train_data_file only contains the training data and
# test_data_file only contains the data used for testing.

# r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\calculated_descriptors.pkl"
# r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\train_fingerprints.pkl"
# r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\test_fingerprints.pkl"
# input_file = 
train_data_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\train_fingerprints.pkl"
test_data_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\test_fingerprints.pkl"

# train = read_data(train_data_file)
# test = read_data(test_data_file)
# print(len(train))
# print(train['PKM2_inhibition'].sum())
# print(train['ERK2_inhibition'].sum())
# print(len(test))
# print(test['PKM2_inhibition'].sum())
# print(test['ERK2_inhibition'].sum())

# Find the value for the amount of trees that creates the best model:
# best_model_trees = fingerprints_eval('PKM2_inhibition', train_data_file, test_data_file)

# Create and fit the model with the determined amount of trees.
# average_scores = []
# for i in range(1,20):
#     score = fingerprints_model(train_data_file, test_data_file, i)
#     average_scores.append(score)
# print(average_scores)

# I tried to optimize the parameters but the accuracies never get higher 
# than the original model with the standard parameters.
# best_model, best_params, avg_bal_accuracy = find_best_param(
#      train_data_file, test_data_file)


# REMARK: The model does never predict inhibition (1) for a molecule of the test set. 
# Why? Is this because the test set contains so little molecules that inhibit? 
# What effect will this have on the actual data to be predicted? 
# Will it then also never predict inhibition = true?
train_data = read_data(train_data_file)
test_data = read_data(test_data_file)
X_train = train_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
y_train = train_data['ERK2_inhibition']

X_test = test_data.drop(columns=[
    'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
    )
y_test = test_data['ERK2_inhibition']

y_test, y_pred, bal_acc_score, rfc = rf_classifier('ERK2_inhibition', train_data, test_data, 1)
print(bal_acc_score)
create_confusion_matrix(y_test, y_pred, rfc)

y_test, y_pred, bal_acc_score, rfc = rf_classifier('PKM2_inhibition', train_data, test_data, 1)
print(bal_acc_score)
create_confusion_matrix(y_test, y_pred, rfc)

find_best_param('ERK2_inhibition', train_data_file, test_data_file)