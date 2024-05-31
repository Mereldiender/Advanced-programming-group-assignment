# Loading of packages
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, RDKFingerprint, Descriptors3D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Reading in of data
df_old = pd.read_pickle('Data/train_descriptors.pkl')
df = df_old.reset_index()

# Compute dataframe that only contains features
df_features_only = df.drop(columns=['index', 'PKM2_inhibition', 'ERK2_inhibition', 'SMILES'])
scaler = MinMaxScaler() # adapt scaler if we want
scaled_features_only = scaler.fit_transform(df_features_only) # fit the scaler to the data, then transform it
df_scaled_features_only = pd.DataFrame(scaled_features_only, columns=df_features_only.columns) # format the transformed data  back to dataframe structure

# Perform PCA
pca = PCA() 
pca_features = pca.fit(df_scaled_features_only) 

# we calculate the cumulative variance by the summation of the variance captured per principal component
cumulative_variance = np.cumsum(pca_features.explained_variance_ratio_)

# determine nr of principal components to capture at 75, 80 and 90%
num_components_75_variance = np.argmax(cumulative_variance >= 0.75) + 1 #returns indices, so we do +1 for the nr of principal components
num_components_80_variance = np.argmax(cumulative_variance >= 0.80) + 1 #returns indices, so we do +1 for the nr of principal components
num_components_90_variance = np.argmax(cumulative_variance >= 0.90) + 1 #returns indices, so we do +1 for the nr of principal components

print('Number of PCs to capture 75%', num_components_75_variance)
print('Number of PCs to capture 80%', num_components_80_variance)
print('Number of PCs to capture 90%', num_components_90_variance)

pca_columns_75 = [f'PC{i+1}' for i in range(num_components_75_variance)]
pca_columns_80 = [f'PC{i+1}' for i in range(num_components_80_variance)]
pca_columns_90 = [f'PC{i+1}' for i in range(num_components_90_variance)]

pca_75 = PCA(n_components=num_components_75_variance)
pca_80 = PCA(n_components=num_components_80_variance) 
pca_90 = PCA(n_components=num_components_90_variance) 

pca_transformed_75 = pca_75.fit_transform(df_scaled_features_only)
pca_transformed_80 = pca_80.fit_transform(df_scaled_features_only)
pca_transformed_90 = pca_90.fit_transform(df_scaled_features_only)

scores_75_df = pd.DataFrame(pca_transformed_75, columns=pca_columns_75)
scores_80_df = pd.DataFrame(pca_transformed_80, columns=pca_columns_80)
scores_90_df = pd.DataFrame(pca_transformed_90, columns=pca_columns_90)

df_only_molecule_and_inhibition = df[['SMILES', 'PKM2_inhibition', 'ERK2_inhibition']]

# Combine the columns describing the molecule and their inhibition with the PC values
complete_75_df = pd.concat([df_only_molecule_and_inhibition, scores_75_df], axis=1)
complete_80_df = pd.concat([df_only_molecule_and_inhibition, scores_80_df], axis=1)
complete_90_df = pd.concat([df_only_molecule_and_inhibition, scores_90_df], axis=1)

complete_75_df.to_pickle('Data/train_descriptors_pc_75.pkl')
complete_80_df.to_pickle('Data/train_descriptors_pc_80.pkl')
complete_90_df.to_pickle('Data/train_descriptors_pc_90.pkl')