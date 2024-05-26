import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def read_data(input_file):
    data = pd.read_pickle(infput_file)
    return data

def rf_classifier(data):
        # Declare feature vector (X) and target variable (y)
    X = data.drop(columns=['PKM2_inhibition','SMILES'], axis=1)             # What to do with the fact that we have PKM2 and ERK2 to predict?
    y = data['PKM2_inhibition']

    # Split data into test and train set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    ## ---------------- Random Forest Classifier model with default parameters
    # Instantiate the classifier (uses 10 decision trees on default)
    rfc = RandomForestClassifier(random_state=0)

    # Fit the model
    rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = rfc.predict(X_test)

    # Check accuracy score (turns out to be 0.9810)
    print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


    ## ---------------- Find important features with Random Forest model
    # Create the classifier with n_estimators = 100
    clf = RandomForestClassifier(n_estimators=100, random_state=0)

    # Fit the model to the training set
    clf.fit(X_train, y_train)

    # View the feature scores
    feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return feature_scores

# # Creating a seaborn bar plot
def vis_feature_importance(feature_scores):
    sns.barplot(x=feature_scores, y=feature_scores.index)

    # Add labels to the graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')

    # Add title to the graph
    plt.title("Visualizing Important Features")

    # Visualize the graph
    plt.show()

# ## ---------------- Build the Random Forest model on selected features
# # Declare feature vector and target variable
def optimize_rf(data):
    X = data.drop(columns=[], axis=1) # fill out the names of the features that can be neglected
    y = data['PKM2_inhibition']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    # Instantiate the classifier with n_estimators = 100
    clf = RandomForestClassifier(random_state=0)

    # Fit the model to the training set
    clf.fit(X_train, y_train)

    # Predict on the test set results
    y_pred = clf.predict(X_test)

    # Check accuracy score 
    print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


## ---------------- Confustion Matrix
# Print the Confusion Matrix and slice it into four pieces
def create_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    print('Confusion matrix\n', cm)

    # Plot confustion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=clf.classes_)
    disp.plot()
    plt.show()

## ---------------- Use the wrapping technique proposed in literature
# Heb het nog niet uit kunnen testen, want daar was mijn laptop veel te lang mee bezig dus heb het afgekapt.

# This part of the code executes the k-fold cross-validation. 
# What needs yet to be coded:
# After this iteration, remove the least 
# important fraction of the variables and train another learning machine on the remainders. Again, keep
# recording the CV test predictions. Repeat this removing of a fraction and compute the CV test predictions.
# Aggregate the predictions from all k CV test sets and compute the aggregate error rate at each step down
# in number of variables. Select p' that minimizes the curve of median error rate vs. nr of variables. 
# If p' is selected, train a learning machine now on all the data, producing a ranking of the variables, and
# take only the most important p' variables to input in the final learning machine.
def wrapping(X_train, X_test, y_train, y_test):
    k = 10  # Number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    errors = []

    for train_index, test_index in kf.split(X):
        # Split data into training and validation sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=1)
        model.fit(X_train, y_train)

        # Predict on validation set
        y_pred = model.predict(X_test)

        # Compute the error metric (e.g., Mean Squared Error)
        mse = mean_squared_error(y_test, y_pred)
        print(mse)
        errors.append(mse)

    # Average error across all folds
    average_error = sum(errors) / k
    print(f'Average Mean Squared Error: {average_error}')


# ## ---------------- Using random search to optimize random forest (lower accuracy than first model [0.9729])
def hyperparameter_optimization(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import RandomizedSearchCV
    RSEED = 50

    # Hyperparameter grid
    param_grid = {
        'n_estimators': np.linspace(10, 200).astype(int),                       # Amount of trees
        'max_depth': [None] + list(np.linspace(3, 20).astype(int)),             # Maximum number of levels in tree
        'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),  # Number of features to consider at every split
        'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),  # How many leaf nodes can be visited
        'min_samples_split': [2, 5, 10],                                        # Minimum number of samples required to split a node
        'bootstrap': [True, False]                                              # Method of selecting samples for training each tree
    }

    # Estimator for use in random search
    estimator = RandomForestClassifier(random_state = RSEED)

    # Create the random search model
    # n_iter = number of different combinations to try
    # cv = number of folds used for cross validation
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 
                            scoring = 'roc_auc', cv = 3, 
                            n_iter = 10, verbose = 1, random_state=RSEED)

    # Fit 
    rs.fit(X_train, y_train)
    best_model = rs.best_estimator_
    prediction = best_model.predict(X_test)
    print('Model accuracy score after hyperparameter tuning : {0:0.4f}'. format(accuracy_score(y_test, prediction)))


input_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\calculated_descriptors.pkl"