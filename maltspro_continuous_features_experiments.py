import pymaltspro as pmp
from pymaltspro import sample_quantile
from pymaltspro2 import linbo_ITE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, getopt
import pickle as pkl
from multiprocessing import Pool
import time

# def barycenter_imputation(pmp_self, X_estimation, Y_estimation, MG):
#     Y_counterfactual = []
#     for i in X_estimation.index.values:
#         # make a holder list for adding matched units' outcomes
#         matched_unit_ids = MG.query(f'unit == {i}').query(pmp_self.treatment + ' != unit_treatment').matched_unit.values
#         matched_unit_outcomes = Y_estimation[matched_unit_ids, :]
#         y_i_counterfactual = pmp.wasserstein2_barycenter(
#             sample_array_1_through_n = matched_unit_outcomes, 
#             weights = np.repeat(1/matched_unit_outcomes.shape[0], matched_unit_outcomes.shape[0]),
#             n_samples_min=pmp_self.n_samples_min,
#             qtl_id = False
#         )
#         Y_counterfactual.append(y_i_counterfactual)
#     return np.array(Y_counterfactual)

def maltspro_parallel(dataset_iteration):
    print(dataset_iteration)
    seed = 2020 + 1000 * dataset_iteration

    # read dataset
    maltspro_df = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X.csv')
    y = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/Y.csv').to_numpy()

    n_units = maltspro_df.shape[0]

    np.random.seed(999)
        
    # split into training and estimation datasets: 20% for training, 80% for estimation
    train_indexes = np.random.choice(range(n_units), size = int(0.2 * n_units), replace = False)
    est_indexes = list(set(range(n_units)) - set(train_indexes))
    X_train = maltspro_df.iloc[train_indexes, :].reset_index()
    X_est = maltspro_df.iloc[est_indexes, :].reset_index()
    y_train = y[train_indexes, :]
    y_est = y[est_indexes, :]


    # run MALTSPro
    # print('initializing...', time.time())
    # maltspro = pmp.pymaltspro(X = X_train,
    #                           y = y_train, 
    #                           treatment = 'A', 
    #                           discrete = [],
    #                           C = 0.001,
    #                           k = 10)

    # print('fitting...', time.time())
    # print(maltspro.fit(method = 'SLSQP'))
    # print('finished fitting...', time.time())
    
    # # save maltspro
    # pkl_file = open(dataset_directory + '/dataset_' + str(seed) + '/malts_model.pkl', 'wb')
    # pkl.dump(maltspro, file = pkl_file)

    with open(dataset_directory + '/dataset_' + str(seed) + '/malts_model.pkl', 'rb') as f:
        maltspro = pkl.load(f)
    print(dataset_iteration, 'getting matched groups')
    # get matched groups
    mg_df = maltspro.get_matched_groups(X_estimation=X_est,
                                        Y_estimation= y_est,
                                        k =10)

    # print(dataset_iteration, 'getting CATEs')
    # # estimate CATEs with MALTS
    # CATE_malts = maltspro.CATE(X_estimation = X_est,
    #                     Y_estimation = y_est,
    #                     reference_distribution = np.array([np.linspace(0, 1, maltspro.n_samples_min), np.linspace(0, 1, maltspro.n_samples_min)]),
    #                     MG = mg_df)

    # pd.DataFrame(CATE_malts, columns=np.linspace(0, 1, CATE_malts.shape[1])).\
    #         to_csv(dataset_directory + '/dataset_' + str(seed) + '/maltspro_ITE.csv')

    print(dataset_iteration, 'getting ITE')
    y_bary = maltspro.barycenter_imputation(X_estimation = X_est,
                                            Y_estimation = y_est,
                                            MG = mg_df, 
                                            qtl_id = False)
    
    ITE_maltspro = []
    for i in range(len(est_indexes)):
        ITE_maltspro.append(
            linbo_ITE(
                y_obs = y_est[i, :],
                y_cf = y_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                reference_distribution = np.vstack([np.linspace(0, 1, maltspro.n_samples_min),
                                                    np.linspace(0, 1, maltspro.n_samples_min)]),
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )[1, :]
        )
    ITE_maltspro = np.array(ITE_maltspro)
    
    maltspro_ITE_df = pd.DataFrame(ITE_maltspro, columns = np.linspace(0, 1, ITE_maltspro.shape[1]))
    maltspro_ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/maltspro_ITE.csv')

    print('done', dataset_iteration)
    del(maltspro)
    del(mg_df)
    del(ITE_malts)
    
    

if __name__ == '__main__':
    dataset_directory = './experiments/quadratic_sim_dgp'
    
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(maltspro_parallel,
                 dataset_iterations_to_conduct)
