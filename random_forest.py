import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_pickle(r"C:\Users\20212072\OneDrive - TU Eindhoven\Documents\Year3(2023-2024)\Kwartiel4\8CC00 - Advanced programming and biomedical data analysis\Group Assignment\calculated_descriptors.pkl")

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

# Check accuracy score
print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


## ---------------- Find important features with Random Forest model
# Create the classifier with n_estimators = 100
clf = RandomForestClassifier(n_estimators=100, random_state=0)

# Fit the model to the training set
clf.fit(X_train, y_train)

# View the feature scores
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# # Creating a seaborn bar plot
# sns.barplot(x=feature_scores, y=feature_scores.index)

# # Add labels to the graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')

# # Add title to the graph
# plt.title("Visualizing Important Features")

# # Visualize the graph
# plt.show()

# ## ---------------- Build the Random Forest model on selected features
# # Declare feature vector and target variable
# X = data.drop(columns=[], axis=1)
# y = data['PKM2_inhibition']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# # Instantiate the classifier with n_estimators = 100
# clf = RandomForestClassifier(random_state=0)

# # Fit the model to the training set
# clf.fit(X_train, y_train)

# # Predict on the test set results
# y_pred = clf.predict(X_test)

# # Check accuracy score 
# print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


## ---------------- Confustion Matrix
# Print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
print('Confusion matrix\n\n', cm)

# Plot confustion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()