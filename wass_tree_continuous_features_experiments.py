import pandas as pd
import numpy as np
import wasserstein_trees as wstree
from pymaltspro2 import sample_quantile, ITE, linbo_ITE
import pickle as pkl
from multiprocessing import Pool
import sys, getopt

def wass_tree_meta_predict(X_valid, wass_tree_treated, wass_tree_control, treatment_var):
    y_pred = []
    for i in range(X_valid.shape[0]):
        if X_valid.loc[i, treatment_var] == 1:
            y_pred.append(wass_tree_control.predict(X_valid.loc[i:i, :].assign(treatment = 0))[0, :])
        else:
            y_pred.append(wass_tree_treated.predict(X_valid.loc[i:i, :].assign(treatment = 1))[0, :])
    y_pred = np.array(y_pred)
    return y_pred

### Wasserstein random forest experiments
def wass_tree_parallel(dataset_iteration):
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

    # split data into treated and control
    # train control random forest
    wass_tree_control = wstree.wass_forest(X = X_train.query('A == 0'), 
                    y = y_train[X_train.query('A == 0').index.values, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=5,
                    depth=None,
                    node_type=None,
                    n_trees=1,
                    seed=999,
                    n_samples_min=None)

    # train treated random forest
    wass_tree_treated = wstree.wass_forest(X = X_train.query('A == 1'), 
                    y = y_train[X_train.query('A == 1').index.values, :],
                    y_quantile_id=False,
                    min_samples_split=None,
                    max_depth=5,
                    depth=None,
                    node_type=None,
                    n_trees=1,
                    seed=999,
                    n_samples_min=None)

    # save random forest models
    control_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wass_treeControl.pkl', 'wb')
    pkl.dump(wass_tree_control, control_file_name)
    treated_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wass_treeTreated.pkl', 'wb')
    pkl.dump(wass_tree_treated, treated_file_name)

    # impute counterfactuals
    y_wass_tree_bary = wass_tree_meta_predict(X_est, wass_tree_control=wass_tree_control, wass_tree_treated=wass_tree_treated, treatment_var='A')

    # measure P(A > B | A ~ Y_i(1), B ~ Y_i(0)) for units i in estimation set
    # ITE_wass_tree = []
    # for i in range(len(est_indexes)):
    #     ITE_wass_tree.append(
    #         ITE(n_samples_min = wass_tree_control.n_samples_min,
    #             y_true = y_est[i, :],
    #             y_impute = y_wass_tree_bary[i, :],
    #             n_mc_samples = 10000,
    #             obs_treatment = X_est.loc[i, 'A'],
    #             y_true_qtl_id = False,
    #             y_impute_qtl_id = True) # the cf is qtl function of barycenter
    #         )
    # ITE_wass_tree = np.array(ITE_wass_tree)
    
    # # add ITE to ITE dataset
    # ITE_df = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE.csv')
    # ITE_df['ITE_wass_tree'] = ITE_wass_tree
    # print(dataset_iteration, 'ATE_true - ATE_wasstree', ITE_df.ITE_true.mean() - ITE_df.ITE_wass_tree.mean())
    # ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE.csv', index = False)
    
    # estimate the ITE with imputed counterfactuals
    ITE_wass_tree = []
    for i in range(len(est_indexes)):
        ITE_wass_tree.append(
            linbo_ITE(
                y_obs = y_est[i, :],
                y_cf = y_wass_tree_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                reference_distribution = np.vstack([np.linspace(0, 1, wass_tree_control.n_samples_min),
                                                    np.linspace(0, 1, wass_tree_control.n_samples_min)]),
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )[1, :]
        )
    ITE_wass_tree = np.array(ITE_wass_tree)
    ATE_wass_tree = ITE_wass_tree.mean(axis = 1)
    print(dataset_iteration, ': WTree ATE min =', ATE_wass_tree.min(), ' ATE max = ', ATE_wass_tree.max())
    
    wass_tree_ITE_df = pd.DataFrame(ITE_wass_tree, columns = np.linspace(0, 1, ITE_wass_tree.shape[1]))
    wass_tree_ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/wass_tree_ITE.csv')

if __name__ == '__main__':
    dataset_directory = './experiments/quadratic_sim_dgp'
    # opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    # for opt, arg in opts:
    #   if opt == '-f':
    #     dataset_directory = arg 
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(wass_tree_parallel,
                 dataset_iterations_to_conduct)