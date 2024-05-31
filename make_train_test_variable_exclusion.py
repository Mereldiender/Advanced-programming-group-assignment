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


def identify_redundant_variables(significant_pairs, plot=False):
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

    if plot:
        # Compute node sizes based on degree (number of connections)
        degrees = dict(G.degree())
        node_sizes = [degrees[node] * 100 for node in G.nodes()]
        # Draw the graph with increased spacing
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, iterations=50)  # Increase k for more spacing
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.7)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.1)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

        plt.title('Significant Correlation Pairs Graph')
        plt.show()

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

data_set_freshness = 'fresh'    # Can be set to fresh, old, and custom. When fresh, the data set is loaded againg, with old 
                                # you can set your own preferences below.
data_path = r'Data\tested_molecules.csv'
save_path_template = r'Data\raw_data_'
variance_threshold = 10.0
correlation_threshold = 0.9
percentage_test_data = 0.1

# Make settings for data loading
if data_set_freshness == 'fresh':
    import_and_save_data = True
    recall_saved_correlation_df = False
    recall_saved_mean_and_variance_df = False
elif data_set_freshness == 'old':
    import_and_save_data = False
    recall_saved_correlation_df = True
    recall_saved_mean_and_variance_df = True
elif data_set_freshness == 'custom':
    # Enter your custom settings here
    import_and_save_data = False
    recall_saved_correlation_df = True
    recall_saved_mean_and_variance_df = False


# Load the data or recall it from a saved version
path_fingerprints = save_path_template+'fingerprints.pkl'
path_2d_discriptors = save_path_template+'2d_discriptors.pkl'
if import_and_save_data:
    # Read from the CSV
    data = read_inhibition_data(data_path)
    # Calculate finger prints and 2d discriptors
    df_fingerprints = add_all_descriptors_to_df(data, add_fingerprints=True, add_2d_discriptors=False)
    df_2d_discriptors = add_all_descriptors_to_df(data, add_fingerprints=False, add_2d_discriptors=True)
    # Save the df to a pickle
    df_fingerprints.to_pickle(path_fingerprints)
    df_2d_discriptors.to_pickle(path_2d_discriptors)
else:
    df_fingerprints = pd.read_pickle(path_fingerprints)
    df_2d_discriptors = pd.read_pickle(path_2d_discriptors)

# Determine redundant variables
save_path_correlation = save_path_template+'2d_correlation.pk.'  
if recall_saved_correlation_df:
    correlation = pd.read_pickle(save_path_correlation)
else:
    correlation = df_2d_discriptors.corr(numeric_only=True)
    correlation.to_pickle(save_path_correlation)
pairs = find_significant_correlations(correlation, high_threshold=correlation_threshold, low_threshold=-correlation_threshold)
variables_to_exclude = identify_redundant_variables(pairs)

# Determine non_informative variables
path_2d_discriptors_var_mean = save_path_template+'2d_var_mean.pkl'
if recall_saved_mean_and_variance_df:
    variance_mean_df = pd.read_pickle(path_2d_discriptors_var_mean)
else:
    variance = df_2d_discriptors.var(numeric_only=True)
    mean = df_2d_discriptors.mean(numeric_only=True)
    variance_mean_df = pd.DataFrame({'Descriptor': variance.index, 'Mean': mean.values, 'Variance': variance.values})
    variance_mean_df.to_pickle(path_2d_discriptors_var_mean)

# Calculate the percentage that the variance is of the mean   
variance_mean_df['Percentage variance of mean'] = abs(variance_mean_df['Variance'] / variance_mean_df['Mean']) * 100
variance_mean_df.sort_values(by='Percentage variance of mean', ascending=True, inplace=True)

# Determine the low variable discriptors based on threshold
low_variance_descriptors = variance_mean_df[variance_mean_df['Percentage variance of mean'] < variance_threshold]['Descriptor'].tolist()

# Combine the lists of variables to exclude and exclude them
all_variables_to_exclude = set(variables_to_exclude).union(low_variance_descriptors)
print(all_variables_to_exclude)
filtered_2d_descriptors = df_2d_discriptors.drop(columns=all_variables_to_exclude)

# Make train test split
indices = range(len(filtered_2d_descriptors))
train_indices, test_indices = train_test_split(indices, test_size=percentage_test_data, random_state=42)
train_descriptors = filtered_2d_descriptors.iloc[train_indices]
train_fingerprints = df_fingerprints.iloc[train_indices]
test_descriptors = filtered_2d_descriptors.iloc[test_indices]
test_fingerprints = df_fingerprints.iloc[test_indices]

# Save training and test split
train_descriptors.to_pickle(r'Data\train_descriptors.pkl')
test_descriptors.to_pickle(r'Data\test_descriptors.pkl')
train_fingerprints.to_pickle(r'Data\train_fingerprints.pkl')
test_fingerprints.to_pickle(r'Data\test_fingerprints.pkl')

print("Finished")
