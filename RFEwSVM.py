# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:21:16 2024

@author: 20192547
"""
from sklearn.model_selection import train_test_split
import pandas as pd
#from sklearn import svm
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

df_features_inh = pd.read_pickle("calculated_descriptors.pkl")
#remove smiles from features
df_features= df_features_inh.iloc[:,3:4487]
df_features_filt= df_features.dropna(axis='columns')

x_features = df_features_filt
y_pkm2 = df_features_inh.iloc[:,1]
y_erk2 = df_features_inh.iloc[:,2]
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x_features, y_pkm2, test_size=0.3,random_state=109) # 70% training and 30% test


# #Create a svm Classifier
# clf = SVC(kernel='linear') # Linear Kernel

# #Train the model using the training sets
# clf.fit(X_train, y_train)

# #Predict the response for test dataset
# y_pred = clf.predict(X_test)


#Create a svm Classifier
svc_lin=SVC(kernel='linear')
svm_rfe_model=RFE(estimator=svc_lin)

svm_rfe_model_fit=svm_rfe_model.fit(X_train,y_train)
feat_index = pd.Series(data = svm_rfe_model_fit.ranking_, index = X_train.columns)
signi_feat_rfe = feat_index[feat_index==1].index
print('Significant features from RFE',signi_feat_rfe)

print('Original number of features present in the dataset : {}'.format(df_features_filt.shape[1]))
print()
print('Number of features selected by the Recursive feature selection technique is : {}'.format(len(signi_feat_rfe)))