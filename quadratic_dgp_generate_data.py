import pandas as pd
import os 
import numpy as np
from multiprocessing import Pool
from scipy.special import erfinv
from pymaltspro2 import linbo_ITE
from sklearn.preprocessing import PolynomialFeatures

def make_data(dataset_iteration):
    print(dataset_iteration, end = ' ')
    seed = 2020 + 1000 * dataset_iteration
    np.random.seed(seed)
    n_units = 1000
    n_obs_y = 1001
    X = np.random.uniform(low = -1, high = 1, size = [n_units, 11])
    error_a = np.random.normal(loc = 0, scale = 1, size = n_units)
    error_y = np.random.uniform(low = -0.5, high = 0.5, size = n_units)
    A = np.random.binomial(n = 1, p = 1/(1 + np.exp(-1 * (0.1 * X[:, 0] + 0.05 * X[:, 1] + error_a))), size = n_units)

    poly = PolynomialFeatures(degree = 2)
    poly_features = poly.fit_transform(X[:, 1:5])
    control_mean = 2 * X[:, 1] + 3 * X[:, 2] + 4 * X[:, 3] + 5 * X[:, 4] + 6 * X[:, 5] + error_y
    control_mean = np.array([np.repeat(control_mean[i], repeats = n_obs_y, axis = 0) for i in range(control_mean.shape[0])])
    treated_mean = 10 + 2*(2 * X[:, 1] + 3 * X[:, 2] + 4 * X[:, 3] + 5 * X[:, 4] + 6 * X[:, 5]) + poly_features.sum(axis = 1) + error_y
    treated_mean = np.array([np.repeat(treated_mean[i], repeats = n_obs_y, axis = 0) for i in range(treated_mean.shape[0])])

    qtl_samples = np.random.uniform(low = 0, high = 1, size = [n_units, n_obs_y])
    y0 = (np.sin(np.pi * qtl_samples)/8) * control_mean
    y1 = (np.sin(np.pi * qtl_samples)/8) * treated_mean

    print(dataset_iteration, 'Sample size: ', A.sum()/n_units)

    print('got data')
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
    
    control_mean_est = control_mean[est_indexes, :]
    treated_mean_est = treated_mean[est_indexes, :]
    
    # estimate the ITE with known potential outcomes
    qtl_grid = np.linspace(0, 1, n_obs_y) 
    y_control_true = (np.sin(np.pi * qtl_grid)/8) * control_mean_est
    y_treated_true = (np.sin(np.pi * qtl_grid)/8) * treated_mean_est
    
    ITE_true = y_treated_true - y_control_true
    ATE_true = ITE_true.mean(axis = 1)
    
    print(dataset_iteration, ': True ATE min =', ATE_true.min(), ' ATE max = ', ATE_true.max())
    
    ITE_df = pd.DataFrame(ITE_true, columns = np.linspace(0, 1, n_obs_y))
    ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/ITE.csv', index = False)


if __name__ == '__main__':
    dataset_directory = './experiments/quadratic_sim_dgp'
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(make_data, dataset_iterations_to_conduct)
