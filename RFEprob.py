# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:53:33 2024

@author: 20192547
"""
import sklearn
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV
from matplotlib import pyplot
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
#os.chdir('OneDrive - TU Eindhoven\Documents\JAAR 5\Advanced programming and biomedical data analysis')

# with open('tested_molecules.csv', 'r') as csvfile:
#   # Create a reader object
#   csv_reader = csv.reader(csvfile)
 
#df_inhibition=pd.read_csv('tested_molecules.csv') 
df_features_inh = pd.read_pickle("calculated_descriptors.pkl")
#remove smiles from features
df_features= df_features_inh.iloc[:,3:4487]
df_features_filt= df_features.dropna(axis='columns')

x_features = df_features_filt
y_pkm2 = df_features_inh.iloc[:,1]
y_erk2 = df_features_inh.iloc[:,2]

#%%
# define the method
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
# fit the model
rfe.fit(x_features, y_pkm2)
# transform the data
x_transformed = rfe.transform(x_features)

# Print the selected features
selected_features = rfe.support_
print("Selected features: ", selected_features)

# Print ranking of features
ranking = rfe.ranking_
print("Feature ranking: ", ranking)

# Get the boolean mask of selected features
selected_features = rfe.support_

# Get the ranking of features
ranking = rfe.ranking_

# Plot the feature ranking
plt.figure(figsize=(12, 6))
plt.bar(range(len(ranking)), ranking)
plt.xlabel("Feature Index")
plt.ylabel("Ranking")
plt.title("Ranking of Features by RFE")
plt.show()

# Plot the selected features (importance from decision tree)
importances = rfe.estimator_.feature_importances_

plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances)
plt.xlabel("Selected Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importances from Decision Tree")
plt.show()

# Optional: Print selected features and their indices
selected_indices = np.where(selected_features)[0]
print("Selected feature indices: ", selected_indices)
print("Selected features: ", x_features.columns[selected_indices])

for i,v in enumerate(importances):
 print('Feature: %0d, Score: %.5f' % (i,v))
 
pyplot.bar([x for x in range(len(importances))], importances)
pyplot.show()

#%%
x_features = df_features_filt
y_pkm2 = df_features_inh.iloc[:,1]

min_features_to_select = 1  # Minimum number of features to consider
max_features_to_select = 40
clf = DecisionTreeClassifier()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(x_features, y_pkm2)

print(f"Optimal number of features: {rfecv.n_features_}")

import matplotlib.pyplot as plt
import pandas as pd

cv_results = pd.DataFrame(rfecv.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()

#%%

from sklearn.model_selection import cross_val_score

x_features = df_features_filt
y_pkm2 = df_features_inh.iloc[:, 1]

min_features_to_select = 1  # Minimum number of features to consider
max_features_to_select = 2
clf = DecisionTreeClassifier()
cv = StratifiedKFold(5)

# List to store cross-validation scores
cv_scores = []

# Loop through feature subsets from 1 to max_features_to_select
for n_features in range(min_features_to_select, max_features_to_select + 1):
    rfe = RFE(estimator=clf, n_features_to_select=n_features)
    scores = cross_val_score(rfe, x_features, y_pkm2, cv=cv, scoring='accuracy', n_jobs=2)
    cv_scores.append((n_features, np.mean(scores), np.std(scores)))

# Convert the scores to a DataFrame
cv_results = pd.DataFrame(cv_scores, columns=['n_features', 'mean_test_score', 'std_test_score'])

# Determine the optimal number of features
optimal_num_features = cv_results.loc[cv_results['mean_test_score'].idxmax()]['n_features']

print(f"Optimal number of features: {optimal_num_features}")

# Plotting the results
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("RFE with Cross-Validation \nup to max features limit")
plt.axvline(x=max_features_to_select, color='r', linestyle='--', label='Max features limit')
plt.legend()
plt.show()

