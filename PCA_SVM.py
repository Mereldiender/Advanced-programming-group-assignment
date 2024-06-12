# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:41:40 2024

@author: 20192547
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC, SVC

df_pca_incl_inh=pd.read_pickle("train_descriptors_pc_90.pkl")
#remove smiles from datafram and the columns with the inhibition
df_pca= df_pca_incl_inh.iloc[:,3:46]
#y target for pkm2 and erk2
y_pkm2 = np.array(df_pca_incl_inh.iloc[:,1])
y_erk2 = np.array(df_pca_incl_inh.iloc[:,2])

X_train, X_test, y_train, y_test = train_test_split(df_pca, y_pkm2, test_size=0.2, random_state=10)

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("CLASSIFICATION REPORT:")
        print(clf_report)
        print("_______________________________________________")
        print("Confusion Matrix:")
        print(confusion_matrix(y_train, pred))
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("CLASSIFICATION REPORT:")
        print(clf_report)
        print("_______________________________________________")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, pred))
#%%
model_linear = LinearSVC(loss='hinge', dual=True)
model_linear.fit(X_train, y_train)
print_score(model_linear, X_train, y_train, X_test, y_test, train=True)
print_score(model_linear, X_train, y_train, X_test, y_test, train=False)

#%%
# The hyperparameter coef0 controls how much the model is influenced by high degree ploynomials
model_poly = SVC(kernel='poly', degree=2, gamma='auto', coef0=1, C=5)
model_poly.fit(X_train, y_train)
print_score(model_poly, X_train, y_train, X_test, y_test, train=True)
print_score(model_poly, X_train, y_train, X_test, y_test, train=False)

#%%
model_rbf = SVC(kernel='rbf', gamma=0.5, C=0.1)
model_rbf.fit(X_train, y_train)

print_score(model_rbf, X_train, y_train, X_test, y_test, train=True)
print_score(model_rbf, X_train, y_train, X_test, y_test, train=False)

#%%
#penalized SVM to deal with unbalanced dataset
param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100], 
              'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001]}


grid = GridSearchCV(SVC(class_weight='balanced', probability=True), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train, y_train)

best_params = grid.best_params_
print(f"Best params: {best_params}")

svm_clf = SVC(class_weight='balanced', probability=True, **best_params)
svm_clf.fit(X_train, y_train)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)
