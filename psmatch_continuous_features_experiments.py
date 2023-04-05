import pandas as pd
import numpy as np
from pymaltspro2 import sample_quantile, ITE, linbo_ITE
import pickle as pkl
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression

def ps_predict(treatment_query, X_query_ps, treatment_est, y_est, y_est_qtl_id, ps_est, k, n_samples_min):
    '''
    description
    -----------
    find k points with closest treatment propensities to X_query's and compute barycenter of their outcomes
    
    parameters
    ----------
    X_query : vector describing X's covariates
    treatment_query : treatment assignment that we want to impute for
    X_query_ps : propensity score of the query point
    y_est : observed outcome for each unit in estimation set
    y_est_qtl_id : True if y_est is quantile functions
    ps_est : propensity score of each unit in estimation set
    n_samples_min : min number of samples for each outcome vector
    '''
    
    # knn_id = pd.DataFrame({'id' : range(len(ps_est)),
    #                       'treatment' : treatment_est,
    #                       'ps' : ps_est,
    #                       'ps_diff' : np.abs(X_query_ps - ps_est)}).\
    #         sort_values('ps_diff', ascending=True).\
    #         query(f'treatment == {treatment_query}').\
    #         iloc[:k, 0]
    knn_id = np.argsort(np.abs(X_query_ps - ps_est))[:k]
    
    
    
    knn_y = []
    # if outcomes are already quantile functions, just append together
    if y_est_qtl_id == True:
        for i in knn_id:
            knn_y.append(y_est[i, :])
    else:
        quantile_values = np.linspace(start = 0, stop = 1, num = n_samples_min)
        y_est_qtl = np.apply_along_axis(arr = y_est,
                                        axis = 1,
                                        func1d = lambda x: np.quantile(x[x == x], q = quantile_values)
                                        )
        for i in knn_id:
            knn_y.append(y_est_qtl[i, :])
    knn_y = np.array(knn_y)
    y_bary = np.mean(knn_y, axis = 0) # barycenter is col means of quantile functions
    
    return y_bary

### Propensity score matching experiments
def ps_parallel(dataset_iteration):
    print(dataset_iteration)
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
    for col in X_train.columns:
        X_train[col] = X_train[col].values**2
        X_est[col] = X_est[col].values**2

    y_train = y[train_indexes, :]
    y_est = y[est_indexes, :]

    print('fitting prop score...', dataset_iteration)
    # train propensity score model : logistic regression
    #   A ~ X_0^2+ ... + X_p^2
    ps_model = LogisticRegression(penalty = 'none')
    ps_model.fit(X = X_train.drop('A', 1).to_numpy(), y = X_train['A'].values)
    ps_values = ps_model.predict(X_est.drop('A',1).to_numpy())
    
    n_samples_min = np.apply_along_axis(arr = y_est,
                                    axis = 1,
                                    func1d = lambda x: x[x == x].shape[0]).min()
    
    print('imputing barycenter...', dataset_iteration)
    # impute counterfactuals 
    y_ps_bary = []
    for i in range(X_est.shape[0]):
        y_ps_bary.append(
            ps_predict(
                treatment_query = 1 - X_est['A'].values[i], # impute counterfactual
                X_query_ps = ps_values[i],
                treatment_est = X_est['A'].values, 
                y_est = y_est, 
                y_est_qtl_id = False, # observed outcomes were not quantile function
                ps_est = ps_values,
                k = 10, # take 10 nearest neighbors
                n_samples_min=n_samples_min
                )
        )
    y_ps_bary = np.array(y_ps_bary)
    
    print('fitting ITE...', dataset_iteration)
    
    # measure P(A > B | A ~ Y_i(1), B ~ Y_i(0)) for units i in estimation set
    ITE_ps = []
    for i in range(len(est_indexes)):
        ITE_ps.append(
            linbo_ITE(
                y_obs = y_est[i, :],
                y_cf = y_ps_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                reference_distribution = np.vstack([np.linspace(0, 1, n_samples_min), np.linspace(0, 1, n_samples_min)]),
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )[1, :]
            )
    ITE_ps = np.array(ITE_ps)
    ATE_ps = ITE_ps.mean(axis = 1)
    # print(dataset_iteration, ': MALTSPro ATE =', ATE_malts)
    pd.DataFrame(ITE_ps, columns=np.linspace(0, 1, ITE_ps.shape[1])).\
        to_csv(dataset_directory + '/dataset_' + str(seed) + '/ps_ITE.csv')
    
    # add ITE to ITE dataset
    # ITE_df = pd.read_csv(dataset_directory + 'dataset_' + str(seed) + '/ITE.csv')
    # ITE_df['ITE_ps'] = ITE_ps
    # print(dataset_iteration, 'ATE_true - ATE_ps', ITE_df.ITE_true.mean() - ITE_df.ITE_ps.mean())
    # ITE_df.to_csv(dataset_directory + 'dataset_' + str(seed) + '/ITE.csv', index = False)
if __name__ == '__main__':
    dataset_directory = './experiments/quadratic_sim_dgp'
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 100) as pool:
        pool.map(ps_parallel, dataset_iterations_to_conduct)
