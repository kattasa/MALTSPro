import pandas as pd
import numpy as np
import wasserstein_trees as wstree
import pickle as pkl

def wrf_meta_predict(X_valid, wrf_treated, wrf_control, treatment_var):
    y_pred = []
    for i in range(X_valid.shape[0]):
        if X_valid.loc[i, treatment_var] == 1:
            y_pred.append(wrf_control.predict(X_valid.loc[i:i, :].assign(treatment = 0))[0, :])
        else:
            y_pred.append(wrf_treated.predict(X_valid.loc[i:i, :].assign(treatment = 1))[0, :])
    y_pred = np.array(y_pred)
    return y_pred
def sample_quantile(quantile_fn, quantile):
    '''
    description
    -----------
    linearly interpolate quantile function and return value of a given quantile
    
    parameters
    ----------
    quantile_fn : numpy array with values of quantile function at specified quantiles
    quantile : value of quantile
    n_qtls : size of quantile function
    
    returns
    -------
    quantile function evaluated at specified quantile
    '''
    n_qtls = quantile_fn.shape[0] - 1
    quantile_index = quantile * n_qtls
    quantile_floor = int(np.floor(quantile_index))
    quantile_ceil  = int(np.ceil(quantile_index))
    if quantile_floor == quantile_ceil == quantile_index:
        return(quantile_fn[quantile_floor])
    else:
        return np.sum([quantile_fn[quantile_floor] * (quantile_index - quantile_floor), quantile_fn[quantile_ceil] * (quantile_ceil - quantile_index)])
def ITE(n_samples_min, y_true, y_impute, n_mc_samples, obs_treatment, y_true_qtl_id = False, y_impute_qtl_id = False):
    if y_true_qtl_id == False:
        y_true_qtl_fn = np.quantile(y_true, q = np.arange(n_samples_min)/n_samples_min)
    else:
        y_true_qtl_fn = y_true
    
    if y_impute_qtl_id == False:
        y_impute_qtl_fn = np.quantile(y_impute, q = np.arange(n_samples_min)/n_samples_min)
    else:
        y_impute_qtl_fn = y_impute
    
    qtls_to_sample = np.random.uniform(low = 0, high = 1, size = n_mc_samples)
    if obs_treatment == 1:
        y_treat = np.array([sample_quantile(quantile_fn = y_true_qtl_fn, quantile = q) for q in qtls_to_sample])
        y_control = np.array([sample_quantile(quantile_fn = y_impute_qtl_fn, quantile = q) for q in qtls_to_sample])
    else:
        y_treat = np.array([sample_quantile(quantile_fn = y_impute_qtl_fn, quantile = q) for q in qtls_to_sample])
        y_control = np.array([sample_quantile(quantile_fn = y_true_qtl_fn, quantile = q) for q in qtls_to_sample])
        
    return (y_treat > y_control).mean()

### Wasserstein random forest experiments
def wrf_parallel(dataset_iteration):
    print(dataset_iteration, end = ' ')
    seed = 2020 + 1000 * dataset_iteration

    # read dataset
    maltspro_df = pd.read_csv('./experiments/cont_features/dataset_' + str(seed) + '/X.csv')
    y = pd.read_csv('./experiments/cont_features/dataset_' + str(seed) + '/Y.csv').to_numpy()

    n_units = maltspro_df.shape[0]
    # make training, estimation split
    # split into training and estimation datasets: 20% for training, 80% for estimation
    train_indexes = np.random.choice(range(n_units), size = int(0.2 * n_units), replace = False)
    est_indexes = list(set(range(n_units)) - set(train_indexes))

    X_train = maltspro_df.iloc[train_indexes, :].reset_index().drop('index', axis = 1)
    X_est = maltspro_df.iloc[est_indexes, :].reset_index().drop('index', axis = 1)

    y_train = y[train_indexes, :]
    y_est = y[est_indexes, :]

    # split data into treated and control
    # train control random forest
    wrf_control = wstree.wass_forest(X = X_train.query('A == 0'), 
                    y = y_train[X_train.query('A == 0').index.values, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=20,
                    depth=None,
                    node_type=None,
                    n_trees=1,
                    seed=999,
                    n_samples_min=None)

    # train treated random forest
    wrf_treated = wstree.wass_forest(X = X_train.query('A == 1'), 
                    y = y_train[X_train.query('A == 1').index.values, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=20,
                    depth=None,
                    node_type=None,
                    n_trees=1,
                    seed=999,
                    n_samples_min=None)

    # save random forest models
    control_file_name = open('./experiments/cont_features/dataset_' + str(seed) + '/wrfControl.pkl', 'wb')
    pkl.dump(wrf_control, control_file_name)
    treated_file_name = open('./experiments/cont_features/dataset_' + str(seed) + '/wrfTreated.pkl', 'wb')
    pkl.dump(wrf_treated, treated_file_name)

    # impute counterfactuals
    y_wrf_bary = wrf_meta_predict(X_est, wrf_control=wrf_control, wrf_treated=wrf_treated, treatment_var='A')

    # measure P(A > B | A ~ Y_i(1), B ~ Y_i(0)) for units i in estimation set
    ITE_wrf = []
    for i in range(len(est_indexes)):
        ITE_wrf.append(
            ITE(n_samples_min = wrf_control.n_samples_min,
                y_true = y_est[i, :],
                y_impute = y_wrf_bary[i, :],
                n_mc_samples = 10000,
                obs_treatment = X_est.loc[i, 'A'],
                y_true_qtl_id = False,
                y_impute_qtl_id = False)
            )
    ITE_wrf = np.array(ITE_wrf)
    # add ITE to ITE dataset
    ITE_df = pd.read_csv('./experiments/cont_features/dataset_' + str(seed) + '/ITE.csv')
    ITE_df['ITE_wrf'] = ITE_wrf
    ITE_df.to_csv('./experiments/cont_features/dataset_' + str(seed) + '/ITE.csv')
if __name__ == '__main__':
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 40) as pool:
        pool.map(maltspro_parallel, dataset_iterations_to_conduct)
