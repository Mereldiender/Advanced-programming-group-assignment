import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, RDKFingerprint, Descriptors3D
import numpy as np

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

def calculate_descriptors(mol):
    """
    Calculates molecular descriptors for a given RDKit molecule.

    Parameters:
        mol (rdkit.Chem.Mol): An RDKit molecule object.

    Returns:
        dict: A dictionary of descriptor names and their calculated values.
    """
    descriptors = {}
    for name, function in Descriptors.descList:
        try:
            descriptors[name] = function(mol)
        except:
            descriptors[name] = None
    return descriptors

def calculate_fingerprints(mol):
    """
    Calculates molecular fingerprints for a given RDKit molecule.

    Parameters:
        mol (rdkit.Chem.Mol): An RDKit molecule object.

    Returns:
        dict: A dictionary of fingerprint names and their values.
    """
    fingerprints = {}

    # Morgan Fingerprints
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fingerprints.update({f'Morgan_{i}': bit for i, bit in enumerate(morgan_fp)})

    # RDKit Fingerprints
    rdkit_fp = RDKFingerprint(mol)
    fingerprints.update({f'RDKit_{i}': bit for i, bit in enumerate(rdkit_fp)})

    # MACCS Keys
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    fingerprints.update({f'MACCS_{i}': bit for i, bit in enumerate(maccs_fp)})

    return fingerprints

def generate_3d_conformers(mol):
    """
    Generates 3D conformers for a given RDKit molecule.

    Parameters:
        mol (rdkit.Chem.Mol): An RDKit molecule object.

    Returns:
        mol: An RDKit molecule with 3D coordinates or None if the conformers could not be generated.
    """
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return None
    if AllChem.UFFOptimizeMolecule(mol) != 0:
        return None
    return mol

def calculate_3d_descriptors(mol):
    """
    Calculates 3D descriptors for a given RDKit molecule.

    Parameters:
        mol (rdkit.Chem.Mol): An RDKit molecule object with 3D coordinates.

    Returns:
        dict: A dictionary of 3D descriptor names and their calculated values.
    """
    descriptors = {}
    for name, function in Descriptors3D.descList:
        try:
            descriptors[name] = function(mol)
        except:
            descriptors[name] = np.nan
    return descriptors

def add_all_descriptors_to_df(df, add_fingerprints=True, add_2d_discriptors=True):
    """
    Adds all molecular descriptors and fingerprints to the DataFrame.

    Parameters:
        df (pd.DataFrame): The original DataFrame containing the 'SMILES' column.

    Returns:
        pd.DataFrame: A new DataFrame with the original columns and the calculated descriptors and fingerprints.
    """
    all_data = []
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            all_data.append({**{name: np.nan for name, _ in Descriptors.descList},
                             **{name: np.nan for name in range(2048)},  # Morgan Fingerprints
                             **{name: np.nan for name in range(2048)},  # RDKit Fingerprints
                             **{name: np.nan for name in range(166)},   # MACCS Keys
                             **{name: np.nan for name, _ in Descriptors3D.descList}})
            continue
        if add_2d_discriptors:
            # 2D Descriptors
            descriptors_2d = calculate_descriptors(mol)

        if add_fingerprints:
            # Fingerprints
            fingerprints = calculate_fingerprints(mol)

        # # 3D Descriptors
        # mol_3d = generate_3d_conformers(mol)
        # if mol_3d:
        #     descriptors_3d = calculate_3d_descriptors(mol_3d)
        # else:
        #     descriptors_3d = {name: np.nan for name, _ in Descriptors3D.descList}
        if add_fingerprints and add_2d_discriptors:
            all_data.append({**descriptors_2d, **fingerprints})
            
        elif add_2d_discriptors:
            all_data.append({**descriptors_2d})
        elif add_fingerprints:
            all_data.append({**fingerprints})
    
    descriptors_df = pd.DataFrame(all_data)
    result_df = pd.concat([df.reset_index(drop=True), descriptors_df.reset_index(drop=True)], axis=1)
    return result_df

# Useage
# data_path = r'Data\tested_molecules.csv' 
# mol_tested = read_inhibition_data(data_path)
# if mol_tested is not None:
#     mol_tested = add_all_descriptors_to_df(mol_tested)
#     print(mol_tested.head())
