# Perform random forest on PCA components


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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

## ---------------- Confustion Matrix
# Print the Confusion Matrix and slice it into four pieces
def create_confusion_matrix(y_test, y_pred, rfc, title):
    cm = confusion_matrix(y_test, y_pred, labels=rfc.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
    disp.plot()
    plt.title(title)
    plt.show()


## ---------------- Using random search to optimize random forest
def hyperparameter_optimization(X_train, X_test, y_train, y_test):
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
    bal_accuracy_score : float
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
    
    # Calculate the average precision score for each target
    bal_acc_score = balanced_accuracy_score(y_test, prediction)
    print('Model balanced accuracy score after hyperparameter tuning: {0:0.4f}'.format(
        bal_acc_score)
        )
    return best_model, best_params, bal_acc_score


def cross_validation_model_evaluation(inhibition, train_data_file):
    """
    Performs cross-validation on the dataset and evaluates the model.

    Parameters
    ----------
    inhibition : str
        The name of the variable to be predicted
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
    # Value to be predicted
    y = data[inhibition]

    kf = KFold(n_splits=5, shuffle=True)

    models_CV = []
    i = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model, best_params, bal_acc_score = hyperparameter_optimization(X_train, X_test, y_train, y_test)
        y_pred = best_model.predict(X_test)
        bal_acc_score = balanced_accuracy_score(y_test, y_pred)

        models_CV.append({'Model {}'.format(i): best_model, 'BAcc': bal_acc_score})
        #create_confusion_matrix(y_test, y_pred, best_model)
        i += 1
    
    return models_CV


def rf_classifier(train_data, test_data, model, kinase):
    """
    Returns the random forest classifier model used to predict the 
    inhibition of certain molecules to kinases (PKM2 or ERK2).

    Parameters
    ----------
    train_data : dataframe
        Part of the input data used to train the model
    test_data : dataframe
        Part of the input data used to test the model
    kinase : string
        Kinase that is inhibited
    model : instance of RandomForestClassifier
        Random forest model


    Returns
    -------
    y_test : dataframe
        The output variables to be predicted in a dataframe
    y_pred : list of lists
        The output predicted by the rf classifier
    bal_acc_score : float
        The balanced accuracy score of the rf classifier
    rfc : RandomForestClassifier
        The random forest classifier model

    """
    # Declare feature vector (X) and target variable (y)

    X_train = train_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )

    X_test = test_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
    
    if kinase == 'PKM2':
        y_train = train_data['PKM2_inhibition']
        y_test = test_data['PKM2_inhibition']
    elif kinase == 'ERK2':
        y_train = train_data['ERK2_inhibition']
        y_test = test_data['ERK2_inhibition']

    rfc = model
    # Fit the model
    rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = rfc.predict(X_test)

    # Calculate the average precision score for each target
    bal_acc_score = balanced_accuracy_score(y_test, y_pred)
    print('Model balanced accuracy score is:', bal_acc_score)
    
    title_confusion_matrix = 'Confusion matrix of inhibition on ' + kinase
    create_confusion_matrix(y_test, y_pred, rfc, '')

    return y_test, y_pred, bal_acc_score, rfc


## ---------------- Call the functions to create the models
# Insert paths to files where input_file contains all the testing and
# training data, train_data_file only contains the training data and
# test_data_file only contains the data used for testing.

train_data_file = 'C:\\Users\\20212435\\Documents\\GitHub\\Group assignment\\Advanced-programming-group-assignment\\train_descriptors_balanced_pc_80_2.pkl'
test_data_file = 'C:\\Users\\20212435\\Documents\\GitHub\\Group assignment\\Advanced-programming-group-assignment\\train_descriptors_balanced_pc_80.pkl'

models_CV_PKM2 = cross_validation_model_evaluation('PKM2_inhibition', train_data_file)
models_CV_ERK2 = cross_validation_model_evaluation('ERK2_inhibition', train_data_file)

print(models_CV_PKM2)

i = 1
for rfc_model in models_CV_PKM2:
    model = rfc_model['Model {}'.format(i)]
    y_test, y_pred, bal_acc_score, rfc = rf_classifier(read_data(train_data_file), read_data(test_data_file), model, 'PKM2')
    i += 1
    
i = 1
for rfc_model in models_CV_ERK2:
    model = rfc_model['Model {}'.format(i)]
    y_test, y_pred, bal_acc_score, rfc = rf_classifier(read_data(train_data_file), read_data(test_data_file), model, 'ERK2')
    i += 1
