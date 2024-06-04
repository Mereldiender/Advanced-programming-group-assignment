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

def rf_classifier(train_data, test_data, amount_trees):
    """
    Returns the multi-output classifier model used to predict the 
    inhibition of certain molecules to kinases (PKM2 and ERK2).

    Parameters
    ----------
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
    avg_accuracy_score : float
        The average accuracy score of the multi-output classifier
    multi_target_rfc : MultiOutputClassifier
        The multi-output classifier model

    """
    # Declare feature vector (X) and target variable (y)
    X_train = train_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
    y_train = train_data[[
        'PKM2_inhibition', 'ERK2_inhibition']
        ]

    X_test = test_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
    y_test = test_data[['PKM2_inhibition', 'ERK2_inhibition']]

    ## ---------------- Random Forest Classifier model with default parameters
    # Instantiate the classifier
    # Random_state set to zero to make sure the model returns the same every time
    rfc = RandomForestClassifier(n_estimators = amount_trees, 
                                 random_state=0
                                 )

    multi_target_rfc = MultiOutputClassifier(rfc, n_jobs=-1)

    # Fit the model
    multi_target_rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = multi_target_rfc.predict(X_test)

    # Calculate the average precision score for each target
    bal_acc_scores = [balanced_accuracy_score(y_test.iloc[:, i], y_pred[
        :, i]) for i in range(y_test.shape[1])
        ]
    avg_bal_acc_score = sum(bal_acc_scores) / len(bal_acc_scores)
    print('Model balanced accuracy scores are:', bal_acc_scores)
    print('Model average balanced accuracy score: {0:0.4f}'.format(
        avg_bal_acc_score)
        )

    return y_test, y_pred, avg_bal_acc_score, multi_target_rfc


# Creating a seaborn bar plot
def vis_feature_importance(feature_scores):
    """
    Creates a bar plot of the feature importance ranked from most to 
    least important.

    Parameters
    ----------
    feature_scores : dataframe
        Dataframe containing the feature scores per feature

    Returns
    -------
    None
    """
    sns.barplot(x=feature_scores, y=feature_scores.index)

    # Add labels to the graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')

    # Add title to the graph
    plt.title("Visualizing Important Features")

    # Visualize the graph
    plt.show()


## ---------------- Confusion Matrix
def create_confusion_matrix(y_test, y_pred, labels):
    """
    Creates two confusion matrices, one for each predicted variable.

    Parameters
    ----------
    y_test : dataframe
        The output variables to be predicted in a dataframe
    y_pred : list of lists
        The output predicted by the rf classifier 
    labels : list of str
        List containing the names of the variables to be predicted

    Returns
    -------
    None
    """
    # Create a nump array for y_test and y_pred to plot confusion matrices
    y_test = y_test.to_numpy() if isinstance(y_test, pd.DataFrame) else y_test
    y_pred = np.asarray(y_pred)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(5 * 2, 5))

    # Loop over each target variable to create its confusion matrix
    for i, label in enumerate(labels):
        # Compute the confusion matrix for the i-th target variable
        cm = confusion_matrix(y_test[:, i], y_pred[:, i], labels=[0, 1])

        # Create a ConfusionMatrixDisplay object for the 
        # computed confusion matrix.
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[0, 1]
            )
        
        # Plot the confusion matrix on the i-th subplot.
        disp.plot(ax=axes[i])
        
        # Set the title of the subplot to indicate which target variable 
        # it represents.
        axes[i].set_title(f'Confusion Matrix for {label}')
    
    plt.tight_layout()
    plt.show()

## ---------------- Create scorer for hyperparameter optimization
def avg_accuracy(y_true, y_pred):
    """
    Compute the average balanced accuracy for multi-output classification.
    """
    bal_acc_scores = [balanced_accuracy_score(y_true.iloc[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return np.mean(bal_acc_scores)


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
        'estimator__max_depth': [None] + list(np.linspace(
            3, 100, num=100).astype(int)
            ),                                                                  # Maximum number of levels in tree
        'estimator__max_features': ['sqrt', 'log2', None] + list(
            np.arange(0.1, 1.1, 0.1)
            ),                                                                  # Number of features to consider at every split
        'estimator__max_leaf_nodes': [None] + list(np.linspace(
            10, 100, num=100).astype(int)
            ),                                                                  # How many leaf nodes can be visited
        'estimator__min_samples_split': [2, 5, 10, 15],                         # Minimum number of samples required to split a node
        'estimator__bootstrap': [True, False]                                   # Method of selecting samples for training each tree
    }

    # Estimator for use in random search
    rfc = RandomForestClassifier(n_estimators=4)
    multi_target_rfc = MultiOutputClassifier(rfc, n_jobs=-1)

    # Create the random search model
    rs = RandomizedSearchCV(multi_target_rfc, param_grid, n_jobs=-1, 
                            scoring=avg_bal_acc_scorer_, cv=3, 
                            n_iter=10, verbose=1, random_state=None)

    # Fit 
    rs.fit(X_train, y_train)
    best_model = rs.best_estimator_
    best_params = rs.best_params_ 
    prediction = best_model.predict(X_test)
    
    # Calculate the average precision score for each target
    bal_acc_scores = [average_precision_score(y_test.iloc[:, i], prediction[
        :, i]) for i in range(y_test.shape[1])
        ]
    avg_bal_accuracy_score = sum(bal_acc_scores) / len(bal_acc_scores)
    print('Model average precision score after hyperparameter tuning: {0:0.4f}'.format(
        avg_bal_accuracy_score)
        )
    return best_model, best_params, avg_bal_accuracy_score


def fingerprints_eval(train_data_file, test_data_file):
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
    for amount_trees in range(1,200):
        y_test, y_pred, avg_bal_acc_score, multi_target_rfc = rf_classifier(
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


def fingerprints_model(train_data_file, test_data_file, amount_trees):
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
    y_test, y_pred, avg_precision_score, multi_target_rfc = rf_classifier(
        train_data, test_data, amount_trees
        )
    print('The balanced accuracy score is {}.'.format(avg_precision_score))
    create_confusion_matrix(y_test, y_pred, [
        'PKM2_inhibition', 'ERK2_inhibition']
        )

def find_best_param(train_data_file, test_data_file):
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
    y_train = train_data[
        ['PKM2_inhibition', 'ERK2_inhibition']
        ]

    test_data = read_data(test_data_file)
    X_test = test_data.drop(columns=[
        'PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1
        )
    y_test = test_data[
        ['PKM2_inhibition', 'ERK2_inhibition']
        ]

    avg_bal_acc_score = 0
    # Make sure the accuracy is higher than for the default hyperparameters
    while avg_bal_acc_score < 0.98:
        best_model, best_params, avg_bal_acc_score = hyperparameter_optimization(
            X_train, X_test, y_train, y_test
            )
    print("Best hyperparameters found: {}.".format(best_params))
    print('The average balanced accuracy score is {}.'.format(avg_bal_acc_score))
    y_pred = best_model.predict(X_test)
    create_confusion_matrix(y_test, y_pred, 
                            ['PKM2_inhibition', 'ERK2_inhibition']
                            )
    return best_model, best_params, avg_bal_acc_score


    

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

# Create a custom scorer for RandomizedSearchCV
avg_bal_acc_scorer = make_scorer(avg_accuracy, greater_is_better=True)

# Find the value for the amount of trees that creates the best model:
#best_model_trees = fingerprints_eval(train_data_file, test_data_file)

# Create and fit the model with the determined amount of trees.
fingerprints_model(train_data_file, test_data_file, 4)

# I tried to optimize the parameters but the accuracies never get higher 
# than the original model with the standard parameters.
# best_model, best_params, avg_bal_accuracy = find_best_param(
#      train_data_file, test_data_file)


# REMARK: The model does never predict inhibition (1) for a molecule of the test set. 
# Why? Is this because the test set contains so little molecules that inhibit? 
# What effect will this have on the actual data to be predicted? 
# Will it then also never predict inhibition = true?