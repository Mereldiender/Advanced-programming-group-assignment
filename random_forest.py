import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def read_data(input_file):
    data = pd.read_pickle(input_file)
    return data

def rf_classifier(train_data, test_data, amount_trees):
    # Declare feature vector (X) and target variable (y)
    X_train = train_data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    y_train = train_data['PKM2_inhibition']

    X_test = test_data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    y_test = test_data['PKM2_inhibition']

    ## ---------------- Random Forest Classifier model with default parameters
    # Instantiate the classifier
    rfc = RandomForestClassifier(n_estimators = amount_trees, random_state=0)

    # Fit the model
    rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = rfc.predict(X_test)

    # Calculate the accuracy score
    acc_score = accuracy_score(y_test, y_pred)

    ## ---------------- Find important features with Random Forest model
    # Create the classifier with n_estimators = 100
    clf = RandomForestClassifier(n_estimators=100, random_state=0)

    # Fit the model to the training set
    clf.fit(X_train, y_train)

    # View the feature scores
    feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return y_test, y_pred, acc_score, clf

# Creating a seaborn bar plot
def vis_feature_importance(feature_scores):
    sns.barplot(x=feature_scores, y=feature_scores.index)

    # Add labels to the graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')

    # Add title to the graph
    plt.title("Visualizing Important Features")

    # Visualize the graph
    plt.show()

## ---------------- Confustion Matrix
# Print the Confusion Matrix and slice it into four pieces
def create_confusion_matrix(y_test, y_pred, clf):
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    # Plot confustion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=clf.classes_)
    disp.plot()
    plt.show()


## ---------------- Using random search to optimize random forest (lower accuracy than first model [0.9729])
# I DO NOT USE THIS FUNCTION BUT I DID NOT WANT TO REMOVE IT YET
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


def fingerprints_eval(train_data_file, test_data_file):
    train_data = read_data(train_data_file)
    print("The amount of features is equal to {}.".format(len(train_data.axes[1])))
    test_data = read_data(test_data_file)
    acc_score = 0
    for amount_trees in range(1,20):
        y_test, y_pred, accuracy, clf = rf_classifier(train_data, test_data, amount_trees)
        if accuracy > acc_score:
            acc_score = accuracy
            best_model = [y_test, y_pred, clf, amount_trees]
    print('The accuracy score of the best model is {} and it consists of {} trees.'.format(acc_score, best_model[3]))
    return best_model[3]


def fingerprints_model(train_data_file, test_data_file, amount_trees):
    train_data = read_data(train_data_file)
    test_data = read_data(test_data_file)
    y_test, y_pred, accuracy, clf = rf_classifier(train_data, test_data, amount_trees)
    print('The accuracy score is {}.'.format(accuracy))
    create_confusion_matrix(y_test, y_pred, clf)
    

## ---------------- Call the functions to create the model
input_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\calculated_descriptors.pkl"
train_data_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\train_fingerprints.pkl"
test_data_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\test_fingerprints.pkl"
best_model_trees = fingerprints_eval(train_data_file, test_data_file)
fingerprints_model(train_data_file, test_data_file, best_model_trees)

# So far, the functions can be used to predict PKM2 inhibition. Now I need to elaborate and make sure that we can combine
# predictions for the inhibition of PKM2 and ERK2. 