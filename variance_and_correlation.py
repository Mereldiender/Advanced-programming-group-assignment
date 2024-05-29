from reading_data import add_all_descriptors_to_df, read_inhibition_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def find_significant_correlations(corr_matrix, high_threshold=0.8, low_threshold=-0.8):
    """
    Find pairs of indices with very high positive/negative or zero correlation coefficients.
    Function writen by chatGPT 4o

    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix of size n x n
    high_threshold (float): Threshold for very high positive correlation
    low_threshold (float): Threshold for very high negative correlation

    Returns:
    dict: Dictionary with keys 'positive', 'negative', and 'zero' containing pairs of indices
    """
    positive_pairs = []
    negative_pairs = []
    zero_pairs = []

    # Iterate over the upper triangle of the matrix (excluding the diagonal)
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            coefficient = corr_matrix.iat[i, j]
            if coefficient >= high_threshold:
                positive_pairs.append((corr_matrix.index[i], corr_matrix.columns[j]))
            elif coefficient <= low_threshold:
                negative_pairs.append((corr_matrix.index[i], corr_matrix.columns[j]))
            elif coefficient == 0:
                zero_pairs.append((corr_matrix.index[i], corr_matrix.columns[j]))
    
    return {
        'positive': positive_pairs,
        'negative': negative_pairs,
        'zero': zero_pairs
    }

def save_significant_pairs_table_as_png(corr_matrix, significant_pairs, filename='significant_pairs_table.png'):
    """
    Create and save a table of significant correlation pairs sorted by absolute value of the correlation coefficient.
    Function writen by chatGPT 4o

    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix of size n x n
    significant_pairs (dict): Dictionary with keys 'positive', 'negative', and 'zero' containing pairs of indices
    filename (str): The filename for the saved table image

    Returns:
    None
    """
    pairs = []
    
    for pair_list in significant_pairs.values():
        for (i, j) in pair_list:
            # Avoid duplicate pairs (i, j) and (j, i)
            if (j, i) not in [(x[0], x[1]) for x in pairs]:
                coef = corr_matrix.loc[i, j]
                pairs.append((i, j, coef))
    
    # Create a DataFrame
    pairs_df = pd.DataFrame(pairs, columns=['Variable 1', 'Variable 2', 'Correlation Coefficient'])
    
    # Sort by the absolute value of the correlation coefficient
    pairs_df['Absolute Correlation'] = pairs_df['Correlation Coefficient'].abs()
    pairs_df = pairs_df.sort_values(by='Absolute Correlation', ascending=False).drop(columns='Absolute Correlation')

    # Save DataFrame to a .png file
    fig, ax = plt.subplots(figsize=(10, len(pairs_df) * 0.2 + 1))  # Adjust the figure size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=pairs_df.values, colLabels=pairs_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust the scaling as needed
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def save_top_entries_to_png(df, top_x, filename='top_entries_table.png'):
    """
    Save the top x entries of a DataFrame into a .png file.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data
    top_x (int): The number of top entries to save
    filename (str): The filename for the saved table image

    Returns:
    None
    """
    # Select the top x entries
    top_df = df.head(top_x)

    # Round the DataFrame values to 3 decimal places
    top_df = top_df.round(3)
    
    # Save DataFrame to a .png file
    fig, ax = plt.subplots(figsize=(10, len(top_df) * 0.5 + 1))  # Adjust the figure size as needed
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=top_df.values, colLabels=top_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust the scaling as needed
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Settings
# Importing and saving data
import_and_save_data = False
data_path = r'Data\tested_molecules.csv'
save_path = r'Data\tested_molecules.pkl'

# Correlation
calculate_correlation = False
recall_saved_correlation_df = False
save_path_correlation = r'Data/Correlation_matrix.pkl'

# Variance and mean
calculate_mean_and_variance = True
recall_saved_mean_and_variance_df = True
save_path_mean_and_variance = r'Data/Mean_and_variance.pkl'


# Import data, calculate descriptors
if import_and_save_data:   
    mol_tested = read_inhibition_data(data_path)
    if mol_tested is not None:
        mol_tested = add_all_descriptors_to_df(mol_tested)
        mol_tested.to_pickle(save_path)
    else:
        raise Exception("Import of data not succesfull, check if data is in corrct format")
else:
    # Import data including discriptors from saved file
    mol_tested = pd.read_pickle(save_path)

if calculate_correlation:
    if recall_saved_correlation_df:
        correlation = pd.read_pickle(save_path_correlation)
    else:
        # Pair wise correlation
        correlation = mol_tested.corr(numeric_only=True)
        correlation.to_pickle(save_path_correlation)
        print("Correlation matrix created")

    # Save significant variance table
    pairs = find_significant_correlations(correlation, high_threshold=0.9, low_threshold=-0.9)
    save_significant_pairs_table_as_png(corr_matrix=correlation, significant_pairs=pairs, filename='Test.png')

if calculate_mean_and_variance:
    if recall_saved_correlation_df:
        variance_mean_df = pd.read_pickle(save_path_mean_and_variance)
    else:
        # Create table with mean and variance
        variance = mol_tested.var(numeric_only=True)
        mean = mol_tested.mean(numeric_only=True)
        variance_mean_df = pd.DataFrame({'Discriptor':variance.index, 'Mean': mean.values, 'Variance': variance.values})
        # Save the table
        variance_mean_df.to_pickle(save_path_mean_and_variance)


    variance_mean_df['Percentage variance of mean'] = abs(variance_mean_df['Variance'] / variance_mean_df['Mean'])*100
    variance_mean_df.sort_values(by='Percentage variance of mean', ascending=True, inplace=True)
    save_top_entries_to_png(variance_mean_df, 100, 'Test2.png')
    print(variance_mean_df.head())


print("Finished")
