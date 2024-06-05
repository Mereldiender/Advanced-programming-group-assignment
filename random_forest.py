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


def rf_classifier(inhibition, train_data, test_data, amount_trees):
    """
    Returns the random forest classifier model used to predict the 
    inhibition of certain molecules to kinases (PKM2 or ERK2).

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
        The average accuracy score of the rf classifier
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
def hyperparameter_optimization(X_train, X_test, y_train, y_test, trees):
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
    trees : int
        How many trees are used for the random forest model

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
    rfc = RandomForestClassifier(n_estimators=trees)

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


def cross_validation_model_evaluation(inhibition, train_data_file, trees):
    """
    Performs cross-validation on the dataset and evaluates the model.

    Parameters
    ----------
    inhibition : str
        The name of the variable to be predicted
    train_data_file : str
        The name of the file that contains the data
    trees : int
        How many trees are used for the random forest classifier
    
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

        best_model, best_params, bal_acc_score = hyperparameter_optimization(X_train, X_test, y_train, y_test, trees)
        y_pred = best_model.predict(X_test)
        bal_acc_score = balanced_accuracy_score(y_test, y_pred)

        models_CV.append({'Model {}'.format(i): best_model, 'BAcc': bal_acc_score})
        #create_confusion_matrix(y_test, y_pred, best_model)
        i += 1
    
    return models_CV


## ---------------- Call the functions to create the models
# Insert paths to files where input_file contains all the testing and
# training data, train_data_file only contains the training data and
# test_data_file only contains the data used for testing.

# r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\calculated_descriptors.pkl"
# r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\train_fingerprints.pkl"
# r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\test_fingerprints.pkl"
# input_file = 
train_data_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\train_fingerprints.pkl"
test_data_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\test_fingerprints.pkl"

for i in range(1,20):
#models_PKM2 = cross_validation_model_evaluation('PKM2_inhibition', train_data_file, i)
    models_ERK2 = cross_validation_model_evaluation('ERK2_inhibition', train_data_file, i)
#print(models_PKM2)
    print(models_ERK2)