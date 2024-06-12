# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 19:19:29 2024

@author: 20192547
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt

df_pca_incl_inh=pd.read_pickle("train_descriptors_pc_80.pkl")
#remove smiles from datafram and the columns with the inhibition
df_pca= df_pca_incl_inh.iloc[:,3:46]
#y target for pkm2 and erk2
y_pkm2 = np.array(df_pca_incl_inh.iloc[:,1])
y_erk2 = np.array(df_pca_incl_inh.iloc[:,2])


# For the sake of the example, let's generate some sample data
np.random.seed(42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_pca, y_pkm2, test_size=0.2, random_state=42)#stratify=y_pkm2
# 
#%%
# Compute class weights
class_weights = compute_sample_weight(class_weight='balanced', y=y_train)

model = xgb.XGBClassifier()

#Training the model on the training data
model.fit(X_train, y_train,sample_weight=class_weights)

#Making predictions on the test set
predictions = model.predict(X_test)

#Calculating accuracy
bal_accuracy = balanced_accuracy_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("Balanced Accuracy:", bal_accuracy)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(conf_matrix)

feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

#%%
# Define the XGBoost classifier
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Define the parameter grid for GridSearchCV
# param_grid = {
#     'max_depth': range (2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05],
#     'scale_pos_weight': [1, 10, 20, 30, 40, 50]  # To handle imbalance
# }
param_grid = {
    'max_depth': [4, 5, 6, 7,10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.05, 0.1, 0.2],
    'scale_pos_weight': [1, 10, 20, 30, 40]  # To handle imbalance
}

# Define the k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the GridSearchCV
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='balanced_accuracy', cv=cv, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train, sample_weight=class_weights)

best_params = grid_search.best_params_
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',**best_params)
xgb_clf.fit(X_train, y_train)
# Print the best parameters found by GridSearchCV
#print("Best parameters found: ", grid_search.best_params_)

# Predict on the test set
y_pred = xgb_clf.predict(X_test)

# Evaluate the model
bal_accuracy = balanced_accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
#%%
print("Balanced Accuracy:", bal_accuracy)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

feat_imp = pd.Series(xgb_clf.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()