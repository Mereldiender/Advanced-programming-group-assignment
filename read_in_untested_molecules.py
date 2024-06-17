from reading_data import add_all_descriptors_to_df
import pandas as pd

mol = pd.read_csv(r'Data\untested_molecules-3.csv') 

mol_fingerprints = add_all_descriptors_to_df(mol, add_2d_discriptors=False, add_fingerprints=True)
mol_descriptors = add_all_descriptors_to_df(mol, add_2d_discriptors=True, add_fingerprints=False)

print(mol_descriptors.head())

excluded_variables = ['Chi0n', 'FpDensityMorgan1', 'BCUT2D_LOGPHI', 'fr_benzene', 'fr_phenol_noOrthoHbond', 'fr_Al_OH_noTert', 'Chi4v', 'BCUT2D_CHGHI', 
                      'FpDensityMorgan2', 'NHOHCount', 'Chi0v', 'LabuteASA', 'MaxPartialCharge', 'BCUT2D_MWLOW', 'SlogP_VSA6', 'MaxAbsEStateIndex', 
                      'MinPartialCharge', 'Chi0', 'BalabanJ', 'HeavyAtomCount', 'Chi1v', 'MolMR', 'FractionCSP3', 'fr_Nhpyrrole', 'MaxAbsPartialCharge', 
                      'BCUT2D_LOGPLOW', 'HeavyAtomMolWt', 'fr_ketone_Topliss', 'Chi2n', 'Chi3v', 'Chi1', 'BCUT2D_CHGLO', 'qed', 'Kappa2', 'Kappa1', 
                      'fr_COO2', 'fr_C_O_noCOO', 'Chi3n', 'MinAbsPartialCharge', 'Chi1n', 'Chi2v', 'fr_nitro_arom', 'NumValenceElectrons', 'AvgIpc', 
                      'ExactMolWt', 'Chi4n', 'FpDensityMorgan3', 'SMR_VSA2', 'PKM2_inhibition', 'ERK2_inhibition']

mol_descriptors.drop(excluded_variables, axis=1, inplace=True)

print(mol_descriptors.head())

mol_descriptors.to_pickle('Data/untested_descriptors.pkl')
mol_fingerprints.to_pickle('Data/untested_fingerprints.pkl')