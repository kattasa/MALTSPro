import pandas as pd
import numpy as np
import wasserstein_trees as wstree
from pymaltspro2 import sample_quantile, ITE, linbo_ITE
import pickle as pkl
from multiprocessing import Pool

def wrf_meta_predict(X_valid, wrf_treated, wrf_control, treatment_var):
    y_pred = []
    for i in range(X_valid.shape[0]):
        if X_valid.loc[i, treatment_var] == 1:
            y_pred.append(wrf_control.predict(X_valid.loc[i:i, :].assign(treatment = 0))[0, :])
        else:
            y_pred.append(wrf_treated.predict(X_valid.loc[i:i, :].assign(treatment = 1))[0, :])
    y_pred = np.array(y_pred)
    return y_pred

### Wasserstein random forest experiments
def wrf_parallel(dataset_iteration):
    print(dataset_iteration, end = ' ')
    seed = 2020 + 1000 * dataset_iteration

    # read dataset
    maltspro_df = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X.csv')
    y = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/Y.csv').to_numpy()

    n_units = maltspro_df.shape[0]
    # make training, estimation split
    # split into training and estimation datasets: 20% for training, 80% for estimation
    train_indexes = np.random.choice(range(n_units), size = int(0.2 * n_units), replace = False)
    est_indexes = list(set(range(n_units)) - set(train_indexes))

    X_train = maltspro_df.iloc[train_indexes, :].reset_index().drop('index', axis = 1)
    X_est = maltspro_df.iloc[est_indexes, :].reset_index().drop('index', axis = 1)

    y_train = y[train_indexes, :]
    y_est = y[est_indexes, :]

    print('training control...', dataset_iteration)
    # split data into treated and control
    # train control random forest
    wrf_control = wstree.wass_forest(X = X_train.query('A == 0'), 
                    y = y_train[X_train.query('A == 0').index.values, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=5,
                    depth=None,
                    node_type=None,
                    n_trees=20,
                    seed=999,
                    n_samples_min=None)

    # train treated random forest
    print('training treated...', dataset_iteration)
    wrf_treated = wstree.wass_forest(X = X_train.query('A == 1'), 
                    y = y_train[X_train.query('A == 1').index.values, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=5,
                    depth=None,
                    node_type=None,
                    n_trees=20,
                    seed=999,
                    n_samples_min=None)

    # save random forest models
    control_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wrfControl.pkl', 'wb')
    pkl.dump(wrf_control, control_file_name)
    treated_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wrfTreated.pkl', 'wb')
    pkl.dump(wrf_treated, treated_file_name)
    
    # with open(dataset_directory + '/dataset_' + str(seed) + '/wrfControl.pkl', 'r') as f:
    #     wrf_control = pkl.load(f)
    # with open(dataset_directory + '/dataset_' + str(seed) + '/wrfTreated.pkl', 'r') as f:
    #     wrf_treated = pkl.load(f)

    # impute counterfactuals
    print('imputing barycenters...', dataset_iteration)
    y_wrf_bary = wrf_meta_predict(X_est, wrf_control=wrf_control, wrf_treated=wrf_treated, treatment_var='A')

    # measure P(A > B | A ~ Y_i(1), B ~ Y_i(0)) for units i in estimation set
    # ITE_wrf = []
    # for i in range(len(est_indexes)):
    #     ITE_wrf.append(
    #         ITE(n_samples_min = wrf_control.n_samples_min,
    #             y_true = y_est[i, :],
    #             y_impute = y_wrf_bary[i, :],
    #             n_mc_samples = 10000,
    #             obs_treatment = X_est.loc[i, 'A'],
    #             y_true_qtl_id = False,
    #             y_impute_qtl_id = True) # imputed cf is quantile of barycenter
    #         )
    # ITE_wrf = np.array(ITE_wrf)
    
    # # add ITE to ITE dataset
    # ITE_df = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE.csv')
    # ITE_df['ITE_wrf'] = ITE_wrf
    # ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE.csv', index = False)

    # estimate the ITE with imputed counterfactuals
    print('estimating ITE...', dataset_iteration)
    ITE_wrf = []
    for i in range(len(est_indexes)):
        ITE_wrf.append(
            linbo_ITE(
                y_obs = y_est[i, :],
                y_cf = y_wrf_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                reference_distribution = np.vstack([np.linspace(0, 1, wrf_control.n_samples_min),
                                                    np.linspace(0, 1, wrf_control.n_samples_min)]),
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )[1, :]
        )
    ITE_wrf = np.array(ITE_wrf)
    ATE_wrf = ITE_wrf.mean(axis = 1)
    print(dataset_iteration, ': True ATE min =', ATE_wrf.min(), ' ATE max = ', ATE_wrf.max())
    
    wrf_ITE_df = pd.DataFrame(ITE_wrf, columns = np.linspace(0, 1, ITE_wrf.shape[1]))
    wrf_ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/wrf_ITE.csv')
    
if __name__ == '__main__':
    dataset_directory = './experiments/quadratic_sim_dgp'
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 40) as pool:
        pool.map(wrf_parallel, dataset_iterations_to_conduct)
