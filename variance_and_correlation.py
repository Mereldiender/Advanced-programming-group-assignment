from reading_data import add_all_descriptors_to_df, read_inhibition_data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import tkinter as tk
from tkinter import simpledialog
from sklearn.model_selection import train_test_split


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
    pairs_df = pairs_df.round(3)
    
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

def plot_significant_correlations_graph(significant_pairs):
    """
    Plot a graph for significant correlation pairs with node sizes based on the number of connections
    and print the nodes for each subgraph.

    Parameters:
    significant_pairs (dict): Dictionary with keys 'positive', 'negative', and 'zero' containing pairs of indices

    Returns:
    None
    """
    # Create a graph
    G = nx.Graph()

    # Add edges for positive, negative, and zero correlations
    for correlation_type in ['positive', 'negative', 'zero']:
        for (i, j) in significant_pairs[correlation_type]:
            G.add_edge(i, j)

    # Compute node sizes based on degree (number of connections)
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 100 for node in G.nodes()]

    # Find and print connected components (subgraphs)
    subgraphs = list(nx.connected_components(G))
    for idx, subgraph in enumerate(subgraphs, start=1):
        print(f"Subgraph {idx}: {sorted(subgraph)}")

    # Draw the graph with increased spacing
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, iterations=50)  # Increase k for more spacing
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.title('Significant Correlation Pairs Graph')
    plt.show()

def select_variable_to_keep_with_gui(subgraph):
    """
    Display a pop-up window with a list of variables to keep and exclude, and allow the user to select
    one variable to keep and provide a note for the selection.

    Parameters:
    subgraph (set): A set of variable names in the subgraph

    Returns:
    dict: Dictionary with keys 'keep', 'exclude', and 'note' containing the selected variables and the note
    """
    root = tk.Tk()
    root.title("Select Variable to Keep")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    label = tk.Label(frame, text="Select the variable to keep from the subgraph:")
    label.pack()

    variables = list(subgraph)
    keep_var = tk.StringVar()
    keep_var.set(variables[0])  # Default to the first variable

    for var in variables:
        rb = tk.Radiobutton(frame, text=var, variable=keep_var, value=var)
        rb.pack(anchor='w')

    note_label = tk.Label(frame, text="Add a note for the selection:")
    note_label.pack()
    note_text = tk.Text(frame, height=4, width=40)
    note_text.pack()

    def on_submit():
        selected_keep_var = keep_var.get()
        exclude_vars = [var for var in variables if var != selected_keep_var]
        note = note_text.get("1.0", tk.END).strip()
        result['keep'] = selected_keep_var
        result['exclude'] = exclude_vars
        result['note'] = note
        root.destroy()

    submit_btn = tk.Button(frame, text="Submit", command=on_submit)
    submit_btn.pack(pady=10)

    result = {}
    root.mainloop()

    return result


def identify_redundant_variables(corr_matrix, significant_pairs):
    """
    Identify redundant variables based on correlation and user selection.

    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix of size n x n
    significant_pairs (dict): Dictionary with keys 'positive', 'negative', and 'zero' containing pairs of indices

    Returns:
    set: Set of variables to exclude
    """
    # Create a graph
    G = nx.Graph()

    # Add edges for positive and negative correlations
    for correlation_type in ['positive', 'negative']:
        for (i, j) in significant_pairs[correlation_type]:
            G.add_edge(i, j)

    # Find connected components (subgraphs)
    subgraphs = list(nx.connected_components(G))

    # Identify variables to exclude from each subgraph
    variables_to_exclude = set()
    for subgraph in subgraphs:
        selection = select_variable_to_keep_with_gui(subgraph)
        variables_to_exclude.update(selection['exclude'])

        # Print the node, included and excluded variables, and the note to the terminal
        print(f"Subgraph: {sorted(subgraph)}")
        print(f"Included Variable: {selection['keep']}")
        print(f"Excluded Variables: {selection['exclude']}")
        print(f"Note: {selection['note']}")

    return variables_to_exclude

# Settings
# Importing and saving data
import_and_save_data = True
add_fingerprints = True
add_2d_discriptors = True
data_path = r'Data\tested_molecules.csv'
save_path = r'Data\tested_molecules_no_fingerprints.pkl'

# Correlation
calculate_correlation = True
recall_saved_correlation_df = True
save_path_correlation = r'Data/Correlation_matrix.pkl'

# Variance and mean
calculate_mean_and_variance = True
recall_saved_mean_and_variance_df = False
save_path_mean_and_variance = r'Data/Mean_and_variance.pkl'


# Import data, calculate descriptors
if import_and_save_data:   
    mol_tested = read_inhibition_data(data_path)
    if mol_tested is not None:
        mol_tested = add_all_descriptors_to_df(mol_tested, add_fingerprints=add_fingerprints, add_2d_discriptors=add_2d_discriptors)
        mol_tested.to_pickle(save_path)
    else:
        raise Exception("Import of data not succesfull, check if data is in corrct format")
else:
    # Import data including discriptors from saved file
    mol_tested = pd.read_pickle(save_path)

# calculate correlations, plot significant pairs and select variables to keep
if calculate_correlation:
    if recall_saved_correlation_df:
        correlation = pd.read_pickle(save_path_correlation)
    else:
        # Pairwise correlation
        correlation = mol_tested.corr(numeric_only=True)
        correlation.to_pickle(save_path_correlation)
        print("Correlation matrix created")

    # Find significant pairs
    pairs = find_significant_correlations(correlation, high_threshold=0.9, low_threshold=-0.9)
    plot_significant_correlations_graph(pairs)
    save_significant_pairs_table_as_png(corr_matrix=correlation, significant_pairs=pairs, filename='Test.png')

    # Identify redundant variables
    variables_to_exclude = identify_redundant_variables(correlation, pairs)

variance_threshold = 10.0
if calculate_mean_and_variance:
    if recall_saved_mean_and_variance_df:
        variance_mean_df = pd.read_pickle(save_path_mean_and_variance)
    else:
        print("calculate new mean and variance")
        # Create table with mean and variance
        variance = mol_tested.var(numeric_only=True)
        mean = mol_tested.mean(numeric_only=True)
        variance_mean_df = pd.DataFrame({'Descriptor': variance.index, 'Mean': mean.values, 'Variance': variance.values})
        # Save the table
        variance_mean_df.to_pickle(save_path_mean_and_variance)

    variance_mean_df['Percentage variance of mean'] = abs(variance_mean_df['Variance'] / variance_mean_df['Mean']) * 100
    variance_mean_df.sort_values(by='Percentage variance of mean', ascending=True, inplace=True)
    save_top_entries_to_png(variance_mean_df, 150, 'Test2.png')
    print(variance_mean_df.head())

    low_variance_descriptors = variance_mean_df[variance_mean_df['Percentage variance of mean'] < variance_threshold]['Descriptor'].tolist()

# Combine the lists of variables to exclude
all_variables_to_exclude = set(variables_to_exclude).union(low_variance_descriptors)

# Filter the DataFrame to remove the excluded variables
filtered_mol_tested = mol_tested.drop(columns=all_variables_to_exclude)

# Split the data into training and test sets
fingerprint_columns = [col for col in filtered_mol_tested.columns if 'fingerprint' in col]
descriptor_columns = [col for col in filtered_mol_tested.columns if col not in fingerprint_columns]

fingerprints = filtered_mol_tested[fingerprint_columns]
descriptors = filtered_mol_tested[descriptor_columns]

# Create train-test split
train_descriptors, test_descriptors = train_test_split(descriptors, test_size=0.2, random_state=42)
train_fingerprints, test_fingerprints = train_test_split(fingerprints, test_size=0.2, random_state=42)

print(train_descriptors.head())
print(train_fingerprints.head())
print(all_variables_to_exclude)
# Save the dataframes
train_descriptors.to_pickle('train_descriptors.pkl')
test_descriptors.to_pickle('test_descriptors.pkl')
train_fingerprints.to_pickle('train_fingerprints.pkl')
test_fingerprints.to_pickle('test_fingerprints.pkl')


print("Finished")
