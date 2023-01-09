#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymaltspro as pmp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle as pkl


# In[2]:


def barycenter_imputation(pmp_self, X_estimation, Y_estimation, MG):
    Y_counterfactual = []
    for i in X_estimation.index.values:
        # make a holder list for adding matched units' outcomes
        matched_unit_ids = MG.query(f'unit == {i}').query(pmp_self.treatment + ' != unit_treatment').matched_unit.values
        matched_unit_outcomes = Y_estimation[matched_unit_ids, :]
        y_i_counterfactual = pmp.wasserstein2_barycenter(
            sample_array_1_through_n = matched_unit_outcomes, 
            weights = np.repeat(1/matched_unit_outcomes.shape[0], matched_unit_outcomes.shape[0]),
            n_samples_min=pmp_self.n_samples_min
        )
        Y_counterfactual.append(y_i_counterfactual)
    return np.array(Y_counterfactual)

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
def ITE(pmp_self, y_true, y_impute, n_mc_samples, obs_treatment, y_true_qtl_id = False, y_impute_qtl_id = False):
    if y_true_qtl_id == False:
        y_true_qtl_fn = np.quantile(y_true, q = np.arange(pmp_self.n_samples_min)/pmp_self.n_samples_min)
    else:
        y_true_qtl_fn = y_true
    
    if y_impute_qtl_id == False:
        y_impute_qtl_fn = np.quantile(y_impute, q = np.arange(pmp_self.n_samples_min)/pmp_self.n_samples_min)
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


# #### Simulation Setup: multiple continuous covariates
# - for i = 1...n
#     - $x_{i0},...,x_{i10} \sim Unif[0, 1]$
#     - $error_{a,i} \sim Normal(0, 1)$
#     - $error_{y, i} \sim Normal(0, 1)^2$: $\chi^2$ with 1 df
#     - $a_i = 1(expit(x_{i0} + x_{i1} + error_{a, i}) > 0.5)$
#     - if $a_i = 0$
#         - $$y_i = Beta(sin(pi * x_{i1) * x_{i2}) + 20(x_{i3} - 0.5)^2 + 10x_{i4} + 5x_{i5} + error_{y, i}, 
#                       sin(pi * x_{i1) * x_{i2}) + 20(x_{i3} - 0.5)^2 + 10x_{i4} + 5x_{i5} + error_{y, i})$$
#         - $E_{Y_i(0)}[Y_i(0)] = 1/2$
#     - if $a_i = 1$
#         - $y_i = Exp(1/(0.5 + error_{y, i})) - 1$
#         - $E_{Y_i(1)}[Y_i(1)] = 1/2 + error_y - 1$
#         - $E_i[E_{Y_i(1)}[Y_i(1)] | X] = E_i[1/2 + error_{y,i} - 1] = 1/2 + 1 - 1 = 1/2$

# In[4]:


# create dataset
for dataset_iteration in range(1, 100):
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
        A = (1/(1 + np.exp(-1 * (X[:, 0] + X[:, 1] + error_a))) > 0.5).astype(int)
        y0_param = (np.sin(np.pi * X[:, 1] * X[:, 2]) + 20*(X[:, 3] - 0.5)**2 + 10*X[:, 4] + error_y**2)
#         y1_param = 30 * (np.sin(np.pi * X[:, 1] * X[:, 2]) + 20*(X[:, 3] - 0.5)**2 + 10*X[:, 4] + np.cos(np.pi * X[:, 1] * X[:, 2]) + error_y**2)
        y1_param = 1/(0.5 + error_y)
    y0 = np.array([np.random.beta(a = y0_param[i], b = y0_param[i], size = n_obs_y) for i in range(n_units)])
#     y1 = np.array([np.random.beta(a = y1_param[i], b = y1_param[i], size = n_obs_y) for i in range(n_units)]) - 1
    y1 = np.array([np.random.exponential(scale = y1_param[i], size = n_obs_y) for i in range(n_units)]) - 1
    y = np.array([y0[i, :] if A[i] == 0  else y1[i, :] for i in range(n_units)])
    y_unobs = np.array([y0[i, :] if A[i] == 1  else y1[i, :] for i in range(n_units)])
    maltspro_df = pd.DataFrame(np.hstack([X,A.reshape([n_units, 1])]), columns=list('X_' + str(i) for i in range(11)) + ['A'])

    if 'dataset_' + str(seed) not in os.listdir('./experiments/cont_features/.'):
        os.mkdir('./experiments/cont_features/dataset_' + str(seed))
    maltspro_df.to_csv('./experiments/cont_features/dataset_' + str(seed) + '/X.csv', index = False)
    pd.DataFrame(y).to_csv('./experiments/cont_features/dataset_' + str(seed) + '/Y.csv', index = False)

    np.random.seed(999)
        
    # split into training and estimation datasets: 20% for training, 80% for estimation
    train_indexes = np.random.choice(range(n_units), size = int(0.2 * n_units), replace = False)
    est_indexes = list(set(range(n_units)) - set(train_indexes))
    X_train = maltspro_df.iloc[train_indexes, :].reset_index()
    X_est = maltspro_df.iloc[est_indexes, :].reset_index()
    y_train = y[train_indexes, :]
    y_est = y[est_indexes, :]
    y_unobs_est = y_unobs[est_indexes, :]
    
    # run MALTSPro
    maltspro = pmp.pymaltspro(X = X_train,
                              y = y_train, 
                              treatment = 'A', 
                              discrete = [],
                              C = 0.001,
                              k = 10)

    maltspro.fit(method = 'SLSQP')
        
    # save maltspro
    pkl_file = open('./experiments/cont_features/dataset_' + str(seed) + '/malts_model.pkl', 'wb')
    pkl.dump(maltspro, file = pkl_file)
    
    # get matched groups
    mg_df = maltspro.get_matched_groups(X_estimation=X_est,
                                        Y_estimation= y_est,
                                        k =10)
    
    
    # impute counterfactuals using barycenter of k nn
    y_bary = barycenter_imputation(pmp_self = maltspro, 
                                   X_estimation=X_est,
                                   Y_estimation= y_est,
                                   MG = mg_df)
    
    # estimate the ITE with known counterfactuals
    ITE_true = []
    for i in range(len(est_indexes)):
        ITE_true.append(
            ITE(pmp_self = maltspro, 
                y_true = y_est[i, :],
                y_impute = y_unobs_est[i, :],
                n_mc_samples = 10000,
                obs_treatment = X_est.loc[i, 'A'],
                y_true_qtl_id = False,
                y_impute_qtl_id = False)
            )
    ITE_true = np.array(ITE_true)
    ATE_true = ITE_true.mean()
    print('True ATE:', ATE_true, end = ' ')
    # estimate ITE with MALTS' counterfactual 
    ITE_malts = []
    for i in range(len(est_indexes)):
        ITE_malts.append(
            ITE(pmp_self = maltspro, 
                y_true = y_est[i, :],
                y_impute = y_bary[i, :],
                n_mc_samples = 10000,
                obs_treatment = X_est.loc[i, 'A'],
                y_true_qtl_id = False,
                y_impute_qtl_id = False)
            )
    ITE_malts = np.array(ITE_malts)
    ATE_malts = ITE_malts.mean()
    print('MALTSPro ATE:', ATE_malts)
    
    ITE_df = pd.DataFrame({'ITE_true' : ITE_true, 'ITE_malts' : ITE_malts})
    ITE_df.to_csv('./experiments/cont_features/dataset_' + str(seed) + '/ITE.csv', index = False)
    
    # delete objects locally
    del(seed)
    del(n_units)
    del(n_obs_y)
    del(X)
    del(error_a)
    del(error_y)
    del(A)
    del(y0_param)
    del(y1_param)
    del(y0)
    del(y1)
    del(y)
    del(y_unobs)
    del(train_indexes)
    del(est_indexes)
    del(X_train)
    del(X_est)
    del(y_train)
    del(y_est)
    del(y_unobs_est)
    del(maltspro)
    del(y_bary)
    del(ITE_malts)
    del(ITE_true)
    del(ATE_malts)
    del(ATE_true)
    del(ITE_df)

