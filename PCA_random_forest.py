import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    )

### THIS IS THE SAME CODE AS ROSANNE USED IN random_forest.py

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

    # Calculate the accuracy score for each target
    accuracy_scores = [accuracy_score(
        y_test.iloc[:, i], y_pred[:, i]) for i in range(y_test.shape[1])
        ]
    avg_accuracy_score = sum(accuracy_scores) / len(accuracy_scores)

    return y_test, y_pred, avg_accuracy_score, multi_target_rfc

def forest_eval(train_data_file, test_data_file):
    """
    Returns the amount of trees that leads to the model with the highest 
    average accuracy score.

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
        accuracy score.
    """
    train_data = read_data(train_data_file)
    print("The amount of features is equal to {}.".format(
        len(train_data.axes[1]))
        )
    test_data = read_data(test_data_file)
    bal_acc_score = 0 # balanced accuracy
    
    for amount_trees in range(1,20):
        y_test, y_pred, balanced_accuracy, multi_target_rfc = rf_classifier(
            train_data, test_data, amount_trees
            )
        # Save the values of the best model
        if balanced_accuracy > bal_acc_score:
            acc_score = balanced_accuracy
            best_model = [y_test, y_pred, multi_target_rfc, amount_trees]
    print('The balanced accuracy score of the best model is {} and it consists of {} trees.'.format(
        acc_score, best_model[3])
        )
    return best_model[3]

def train_model(train_df, test_df, amount_trees):
    """
    Create the model with the best amount of trees and plot two confusion 
    matrices.

    Parameters
    ----------
    train_df : dataframe
        Dataframe that contains the training data
    test_df : dataframe
        Dataframe that contains the testing data
    amount_trees : int
        The amount of trees used in the model
    
    Returns
    -------
    None        
    """
    y_test, y_pred, balanced_accuracy, multi_target_rfc = rf_classifier(
        train_df, test_df, amount_trees
        )
    print('The balanced accuracy score is {}.'.format(balanced_accuracy))
    create_confusion_matrix(y_test, y_pred, [
        'PKM2_inhibition', 'ERK2_inhibition']
        )
    print(y_pred)
    
    return multi_target_rfc

def make_predictions(untested_molecules_file, train_data_file):
    """
    Make predictions for untested molecules and write to untested_molecules_file

    Parameters
    ----------
    untested_molecules_file : str
        The name of the CSV file that contains the untested molecules with NaN values
    train_data_file : str
        The name of the PKL file that contains the training data for training the model
    
    """
    # Read the untested molecules file
    untested_molecules = pd.read_csv(untested_molecules_file)
    
    # Create and fit the model with the determined amount of trees
    train_data = read_data(train_data_file)
    
    train_df = train_data.iloc[:600] # create training and testing set
    test_df = train_data.iloc[600:]
    
    best_model_trees = 2 # value was found during testing
    
    rf_model = train_model(train_df, test_df, best_model_trees)
    
    # Extract the features for prediction
    X_untested = untested_molecules.drop(columns=['PKM2_inhibition', 
                                                  'SMILES', 'ERK2_inhibition'], axis=1)
    
    # Generate predictions using the trained model
    y_pred = rf_model.predict(X_untested)
    
    # Fill the NaN values with the predictions
    untested_molecules['PKM2_inhibition'] = y_pred[:, 0]
    untested_molecules['ERK2_inhibition'] = y_pred[:, 1]
    
    # Save the updated dataframe back to the CSV file
    untested_molecules.to_csv(untested_molecules_file, index=False)

## ---------------- Call the functions to create the model
# Insert paths to files where input_file contains all the testing and
# training data, train_data_file only contains the training data and
# test_data_file only contains the data used for testing.

# Make dataframe with all descriptors, here these are principal components
# Initial training set is split into train and test set
#train_descriptors_pc_90_df = read_data(input_file)

#train_df = train_descriptors_pc_90_df.iloc[:600]
#test_df = train_descriptors_pc_90_df.iloc[600:]

train_data_file = 'C:\\Users\\20212435\\Documents\\GitHub\\Group assignment\\Advanced-programming-group-assignment\\train_descriptors.pkl'
#test_data_file = 'C:\\Users\\20212435\\Documents\\GitHub\\Group assignment\\Advanced-programming-group-assignment\\test_descriptors.pkl'
prediction_file = 'C:\\Users\\20212435\\Documents\\GitHub\\Group assignment\\Advanced-programming-group-assignment\\untested_molecules-3.csv'

# Test the make_predictions function:
make_predictions(prediction_file, train_data_file)

"""
What I want is:
Read in file of training, testing and to be predicted data = read_file
Train a model based on training and testing data = train_model
Use the model to predict data from untested molecules = make_predictions
Write untested molecules to file = write_file

Final function: train_file, test_file, prediction_file
"""

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
