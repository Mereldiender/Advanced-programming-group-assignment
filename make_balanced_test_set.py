import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_pickle(r'Data/tested_molecules_raw.pkl')

print('Unplit dataset')
print(df.shape, df['PKM2_inhibition'].sum(), df['ERK2_inhibition'].sum())

# Define the features and targets
X = df.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
y_PKM2 = df['PKM2_inhibition']
y_ERK2 = df['ERK2_inhibition']

# Identify the row where both PKM2_inhibition and ERK2_inhibition are positive
rare_instance = df[(df['PKM2_inhibition'] == 1) & (df['ERK2_inhibition'] == 1)]

# Remove the rare instance from the dataset
df_remaining = df.drop(rare_instance.index)

# Define the features and combined target for the remaining data
X_remaining = df_remaining.drop(columns=['PKM2_inhibition', 'ERK2_inhibition'])
y_combined_remaining = df_remaining['PKM2_inhibition'].astype(str) + df_remaining['ERK2_inhibition'].astype(str)

# Perform the train-test split using the remaining data
X_train, X_test, y_train_combined, y_test_combined = train_test_split(
    X_remaining, y_combined_remaining, test_size=0.1, random_state=42, stratify=y_combined_remaining
)

# Separate the combined target back into individual targets for training and testing
y_train_PKM2 = y_train_combined.apply(lambda x: int(x[0]))
y_train_ERK2 = y_train_combined.apply(lambda x: int(x[1]))
y_test_PKM2 = y_test_combined.apply(lambda x: int(x[0]))
y_test_ERK2 = y_test_combined.apply(lambda x: int(x[1]))


# Convert combined training data back to DataFrame
X_train['PKM2_inhibition'] = y_train_PKM2.values
X_train['ERK2_inhibition'] = y_train_ERK2.values

# Add the rare instance back to the training set
X_train = pd.concat([X_train, rare_instance])

# Convert combined test data back to DataFrame
X_test['PKM2_inhibition'] = y_test_PKM2.values
X_test['ERK2_inhibition'] = y_test_ERK2.values

# Reset indices to avoid any indexing issues
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# Now you have the training and test DataFrames
train_df = X_train
test_df = X_test

# Optionally, verify the contents of the DataFrames
print("Training DataFrame:")
print(train_df.shape, train_df['PKM2_inhibition'].sum(), train_df['ERK2_inhibition'].sum())
print("\nTest DataFrame:")
print(test_df.shape,test_df['PKM2_inhibition'].sum(), test_df['ERK2_inhibition'].sum())

train_df.to_pickle('Data/Trainingset_balanced.pkl')
test_df.to_pickle('Data/Testset_balanced.pkl')