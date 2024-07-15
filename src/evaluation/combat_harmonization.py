from sklearn.impute import KNNImputer
from neuroCombat import neuroCombat, neuroCombatFromTraining

from utils.file_util import load_pickle, save_pickle

def run_combat(data, df_covar, combat_fpath=None, batch_col='Protocol', do_impute=True):
    '''Run combat given data and covariates'''

    # Impute missing values using KNN
    if do_impute:
        imputer = KNNImputer(n_neighbors=3)
        data = imputer.fit_transform(data)

    # Run ComBat
    combat = neuroCombat(dat=data.T, 
                         covars=df_covar,
                         batch_col=batch_col,
                         categorical_cols=['Sex'],
                         continuous_cols=['Age']
                         )
    
    # Save result
    if combat_fpath is not None:
        save_pickle(combat, combat_fpath)

    return combat['data']


def apply_combat(data, batch_values, combat_fpath):
    combat = load_pickle(combat_fpath)
    data_corrected = neuroCombatFromTraining(data.T, batch_values, combat['estimates'])['data']
    return data_corrected.T