from reading_data import add_all_descriptors_to_df
import pandas as pd

desciptors_to_drop = ['Chi0n', 'FpDensityMorgan1', 'BCUT2D_LOGPHI', 'fr_benzene', 'fr_phenol_noOrthoHbond', 'fr_Al_OH_noTert', 'Chi4v', 
                      'BCUT2D_CHGHI', 'FpDensityMorgan2', 'NHOHCount', 'Chi0v', 'LabuteASA', 'MaxPartialCharge', 'BCUT2D_MWLOW', 
                      'SlogP_VSA6', 'MaxAbsEStateIndex', 'MinPartialCharge', 'Chi0', 'BalabanJ', 'HeavyAtomCount', 'Chi1v', 'MolMR', 
                      'FractionCSP3', 'fr_Nhpyrrole', 'MaxAbsPartialCharge', 'BCUT2D_LOGPLOW', 'HeavyAtomMolWt', 'fr_ketone_Topliss', 
                      'Chi2n', 'Chi3v', 'Chi1', 'BCUT2D_CHGLO', 'qed', 'Kappa2', 'Kappa1', 'fr_COO2', 'fr_C_O_noCOO', 'Chi3n', 
                      'MinAbsPartialCharge', 'Chi1n', 'Chi2v', 'fr_nitro_arom', 'NumValenceElectrons', 'AvgIpc', 'ExactMolWt', 
                      'Chi4n', 'FpDensityMorgan3', 'SMR_VSA2']

train = pd.read_pickle(r'Data\Trainingset_balanced.pkl')
train_descriptors = add_all_descriptors_to_df(train, add_2d_discriptors=True, add_fingerprints=False)
train_fingerprints = add_all_descriptors_to_df(train, add_2d_discriptors=False, add_fingerprints=True)
train_descriptors.drop(desciptors_to_drop, axis=1, inplace=True)
train_descriptors.to_pickle('Data/training_descriptors_balanced.pkl')
train_fingerprints.to_pickle('Data/training_fingerprints_balanced.pkl')

test = pd.read_pickle(r'Data\Testset_balanced.pkl')
test_descriptors = add_all_descriptors_to_df(test, add_2d_discriptors=True, add_fingerprints=False)
test_fingerprints = add_all_descriptors_to_df(test, add_2d_discriptors=False, add_fingerprints=True)
test_descriptors.drop(desciptors_to_drop, axis=1, inplace=True)
test_descriptors.to_pickle('Data/test_descriptors_balanced.pkl')
test_fingerprints.to_pickle('Data/test_fingerprints_balanced.pkl')


