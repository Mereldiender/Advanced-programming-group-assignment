import pandas as pd
import rdkit as rd
from rdkit import Chem
from rdkit.Chem import Draw


def read_inhibition_data(csv_file_path):
    """
    Reads a CSV file with columns 'SMILES', 'PKM2_inhibition', and 'ERK2_inhibition'.

    Parameters:
        csv_file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        df = pd.read_csv(csv_file_path, usecols=['SMILES', 'PKM2_inhibition', 'ERK2_inhibition'])
        return df
    except FileNotFoundError:
        print(f"The file {csv_file_path} was not found.")
        return None
    except ValueError as e:
        print(f"Error reading the CSV file: {e}")
        return None

# Example usage:
data_path = r'Data\tested_molecules.csv'
mol_tested = read_inhibition_data(data_path)
example_SMILE = mol_tested['SMILES'].sample().iloc[0]

m = Chem.MolFromSmiles(example_SMILE)
Draw.MolToFile(m, 'Data/test.png')