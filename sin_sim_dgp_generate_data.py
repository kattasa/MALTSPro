import pandas as pd
import os 
import numpy as np
from multiprocessing import Pool
from pymaltspro2 import sample_quantile, ITE, linbo_ITE

def make_data(dataset_iteration):
    print(dataset_iteration, end = ' ')
    seed = 2020 + 1000 * dataset_iteration
    np.random.seed(seed)
    n_units = 1000
    n_obs_y = 1001
    y0_param = np.array([-1])
    y1_param = np.array([-1])
    while sum(y0_param <= 0) != 0 and sum(y1_param <= 0) != 0:
        X = np.random.uniform(low = 0, high = 1, size = [n_units, 11])
        error_a = np.random.normal(loc = 0, scale = 1, size = n_units)
        error_y = np.random.normal(loc = 0, scale = 1, size = n_units)**2 # sample from chi square w/1 df; error_y > 0
        # y0_param = (np.abs(np.sin(np.pi * X[:, 1] * X[:, 2])) + (X[:, 3] - 0.5)**2 + error_y**2)
        # y1_param = (np.abs(np.sin(np.pi * X[:, 1] * X[:, 2])) + 20*(X[:, 3] - 0.5)**2 + np.abs(np.cos(np.pi * X[:, 1] * X[:, 2])) + error_y**2)
        # y1_param = 1/(0.5 + error_y)
        y0_param = np.sin(X[:, 1] + X[:, 2] + error_y) ** 2
        y1_param = np.sin(X[:, 1] + 10 * X[:, 2] + error_y) ** 2
    A = np.random.binomial(n = 1, p = 1/(1 + np.exp(-1 * (X[:, 0] + X[:, 1] + error_a))), size = n_units)
    mixture_id = np.random.binomial(size = n_obs_y, p = 0.25, n = 1)
    y0 = np.array([np.random.beta(a = 2 * y0_param[i], b = 8 * y0_param[i], size = n_obs_y) for i in range(n_units)]) * mixture_id + \
            np.array([np.random.beta(a = 8 * y0_param[i], b = 2 * y0_param[i], size = n_obs_y) for i in range(n_units)]) * (1 - mixture_id)
    y1 = np.array([np.random.beta(a = 2 * y1_param[i], b = 8 * y1_param[i], size = n_obs_y) for i in range(n_units)]) * (1 - mixture_id) + \
            np.array([np.random.beta(a = 8 * y1_param[i], b = 2 * y1_param[i], size = n_obs_y) for i in range(n_units)]) * (mixture_id)
    # y1 = np.array([np.random.exponential(scale = y1_param[i], size = n_obs_y) for i in range(n_units)]) - 1
    y = np.array([y0[i, :] if A[i] == 0  else y1[i, :] for i in range(n_units)])
    y_unobs = np.array([y0[i, :] if A[i] == 1  else y1[i, :] for i in range(n_units)])
    maltspro_df = pd.DataFrame(np.hstack([X,A.reshape([n_units, 1])]), columns=list('X_' + str(i) for i in range(11)) + ['A'])

    if 'dataset_' + str(seed) not in os.listdir(dataset_directory + '/.'):
        os.mkdir(dataset_directory + '/dataset_' + str(seed))
    maltspro_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/X.csv', index = False)
    pd.DataFrame(y).to_csv(dataset_directory + '/dataset_' + str(seed) + '/Y.csv', index = False)
    pd.DataFrame(y_unobs).to_csv(dataset_directory + '/dataset_' + str(seed) + '/Y_unobs.csv', index = False)
    np.random.seed(999)
        
    # split into training and estimation datasets: 20% for training, 80% for estimation
    train_indexes = np.random.choice(range(n_units), size = int(0.2 * n_units), replace = False)
    est_indexes = list(set(range(n_units)) - set(train_indexes))
    X_train = maltspro_df.iloc[train_indexes, :].reset_index()
    X_est = maltspro_df.iloc[est_indexes, :].reset_index()
    y_train = y[train_indexes, :]
    y_est = y[est_indexes, :]
    y_unobs_est = y_unobs[est_indexes, :]
    
    # estimate the ITE with known counterfactuals
    ITE_true = []
    for i in range(len(est_indexes)):
        ITE_true.append(
            linbo_ITE(
                y_obs = y_est[i, :],
                y_cf = y_unobs_est[i, :],
                observed_treatment = X_est['A'].values[i],
                reference_distribution = np.vstack([np.linspace(0, 1, n_obs_y), np.linspace(0, 1, n_obs_y)]),
                y_obs_qtl_id = False, 
                y_cf_qtl_id = False
            )[1, :]
        )
    ITE_true = np.array(ITE_true)
    ATE_true = ITE_true.mean(axis = 1)
    print(dataset_iteration, ': True ATE min =', ATE_true.min(), ' ATE max = ', ATE_true.max())
    
    ITE_df = pd.DataFrame(ITE_true, columns = np.linspace(0, 1, n_obs_y))
    ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE.csv', index = False)
    



if __name__ == '__main__':
    dataset_directory = './experiments/sin_sim_dgp'
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(make_data, dataset_iterations_to_conduct)
