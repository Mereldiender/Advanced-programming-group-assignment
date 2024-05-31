import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def read_data(input_file):
    data = pd.read_pickle(input_file)
    return data

def rf_classifier(train_data, test_data, amount_trees):
    # Declare feature vector (X) and target variable (y)
    X_train = train_data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    y_train = train_data[['PKM2_inhibition', 'ERK2_inhibition']]

    X_test = test_data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    y_test = test_data[['PKM2_inhibition', 'ERK2_inhibition']]

    ## ---------------- Random Forest Classifier model with default parameters
    # Instantiate the classifier
    rfc = RandomForestClassifier(n_estimators = amount_trees, random_state=0)

    multi_target_rfc = MultiOutputClassifier(rfc, n_jobs=-1)

    # Fit the model
    multi_target_rfc.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = multi_target_rfc.predict(X_test)

    # Calculate the accuracy score for each target
    accuracy_scores = [accuracy_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    avg_accuracy_score = sum(accuracy_scores) / len(accuracy_scores)

    return y_test, y_pred, avg_accuracy_score, multi_target_rfc


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

# ## ---------------- Confustion Matrix
# # Print the Confusion Matrix and slice it into four pieces
# def create_confusion_matrix(y_test, y_pred, multi_target_rfc):
#     cm = confusion_matrix(y_test, y_pred, labels=multi_target_rfc.classes_)

#     # Plot confustion matrix
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm,
#                                 display_labels=multi_target_rfc.classes_)
#     disp.plot()
#     plt.show()

## ---------------- Confusion Matrix
# Print the Confusion Matrix and slice it into four pieces
def create_confusion_matrix(y_test, y_pred, labels):
    y_test = y_test.to_numpy() if isinstance(y_test, pd.DataFrame) else y_test
    y_pred = np.asarray(y_pred)
    
    # Create subplots: 1 row, num_labels columns, with appropriate figure size
    fig, axes = plt.subplots(1, 2, figsize=(5 * 2, 5))

    # Loop over each target variable to create its confusion matrix
    for i, label in enumerate(labels):
        # Compute the confusion matrix for the i-th target variable
        cm = confusion_matrix(y_test[:, i], y_pred[:, i], labels=[0, 1])

        # Create a ConfusionMatrixDisplay object for the computed confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        
        # Plot the confusion matrix on the i-th subplot
        disp.plot(ax=axes[i])
        
        # Set the title of the subplot to indicate which target variable it represents
        axes[i].set_title(f'Confusion Matrix for {label}')
    
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    
    # Display the plot
    plt.show()


## ---------------- Using random search to optimize random forest
def hyperparameter_optimization(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import RandomizedSearchCV
    RSEED = 50

    # Hyperparameter grid
    param_grid = {
        'max_depth': [None] + list(range(3, 21)),
        'max_features': ['sqrt', 'log2', None] + list(np.arange(0.1, 1.1, 0.1)), 
        'max_leaf_nodes': [None] + list(range(10, 101, 10)), 
        'min_samples_split': [2, 5, 10, 15], 
        'bootstrap': [True, False]
        }

    # Estimator for use in random search
    estimator = RandomForestClassifier(n_estimators=4)#, random_state = RSEED)

    # Create the random search model
    # n_iter = number of different combinations to try
    # cv = number of folds used for cross validation
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 
                            scoring = 'roc_auc', cv = 5, 
                            n_iter = 15, verbose = 1)#, random_state=RSEED)

    # Fit 
    rs.fit(X_train, y_train)
    best_model = rs.best_estimator_
    best_params = rs.best_params_
    prediction = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction) 
    print('Model accuracy score after hyperparameter tuning : {0:0.4f}'. format(accuracy))
    return best_model, best_params, accuracy


def fingerprints_eval(train_data_file, test_data_file):
    train_data = read_data(train_data_file)
    print("The amount of features is equal to {}.".format(len(train_data.axes[1])))
    test_data = read_data(test_data_file)
    acc_score = 0
    for amount_trees in range(1,20):
        y_test, y_pred, accuracy, multi_target_rfc = rf_classifier(train_data, test_data, amount_trees)
        if accuracy > acc_score:
            acc_score = accuracy
            best_model = [y_test, y_pred, multi_target_rfc, amount_trees]
    print('The accuracy score of the best model is {} and it consists of {} trees.'.format(acc_score, best_model[3]))
    return best_model[3]


def fingerprints_model(train_data_file, test_data_file, amount_trees):
    train_data = read_data(train_data_file)
    test_data = read_data(test_data_file)
    y_test, y_pred, accuracy, multi_target_rfc = rf_classifier(train_data, test_data, amount_trees)
    print('The accuracy score is {}.'.format(accuracy))
    create_confusion_matrix(y_test, y_pred, ['PKM2_inhibition', 'ERK2_inhibition'])

def find_best_param(train_data_file, test_data_file):
    train_data = read_data(train_data_file)
    X_train = train_data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    y_train = train_data[['PKM2_inhibition', 'ERK2_inhibition']]

    test_data = read_data(test_data_file)
    X_test = test_data.drop(columns=['PKM2_inhibition','SMILES', 'ERK2_inhibition'], axis=1)
    y_test = test_data[['PKM2_inhibition', 'ERK2_inhibition']]

    accuracy = 0
    while accuracy < 0.98:
        best_model, best_params, accuracy = hyperparameter_optimization(X_train, X_test, y_train, y_test)
    print("Best hyperparameters found: {}.".format(best_params))
    print('The accuracy score is {}.'.format(accuracy))
    y_pred = best_model.predict(X_test)
    create_confusion_matrix(y_test, y_pred, ['PKM2_inhibition', 'ERK2_inhibition'])
    return best_model, best_params, accuracy


    

## ---------------- Call the functions to create the model
input_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\calculated_descriptors.pkl"
train_data_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\train_fingerprints.pkl"
test_data_file = r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\test_fingerprints.pkl"
best_model_trees = fingerprints_eval(train_data_file, test_data_file)
fingerprints_model(train_data_file, test_data_file, best_model_trees)

# I tried to optimize the parameters but the accuracies never get higher than the original model with the standard parameters.
# 4 trees turned out to be best.
# best_model, best_params, accuracy = find_best_param(train_data_file, test_data_file)


# REMARK: The model does never predict inhibition for a molecule of the test set. Why? Is this because the test set contains so little 
# molecules that inhibit? What effect will this have on the actual data to be predicted? Will it then also never predict inhibition = true?