import pandas as pd
import os 
import numpy as np
from multiprocessing import Pool
from pymaltspro2 import sample_quantile, ITE, linbo_ITE


## tipping for each person is a zero-inflated distribution
## source : https://www.sciencedirect.com/science/article/pii/S221480431500004X#sec0007
## likelihood of tipping/not tipping depends on 

##   age : increase by one year increases tip probability by 0.01
##   education : having a college degree increases probability of tipping by 0
##   income : every additional dollar earned increases probability of tipping by 0.03
##   reward motives : expecting a reward increases probability of tipping by 0.19
##   duty motives : tipping because of societal expectations increases probability of tipping by 0.06
##   altruistic motives : tipping out of one's own goodness increases probability of tipping by 0.14

## non-zero amount tipping depends on
##   age : increase by one year increases tip percent by 0.01
##   education : having a college degree increases tip percent by 0.05
##   income : every additional dollar earned increases tip percent by 0
##   reward motives : expecting a reward increases tip percent by 0.13
##   duty motives : tipping because of societal expectations increases tip percent by -0.06
##   altruistic motives : tipping out of one's own goodness increases tip percent by 0.05

## probability of treatment (getting a coupon) depends on
##   education : having a college degree increases treatment prob of being treated by 0.2
##   income    : having an extra dollar of income increases prob of being treated by 0.1

## treatment 
##   increases probability of tipping by 0.25
##   increases tip percent by 0.1
def make_data(dataset_iteration):
    print(dataset_iteration, end = ' ')
    seed = 2020 + 1000 * dataset_iteration
    np.random.seed(seed)
    
    n_units = 1000
    n_obs_y = 1001
    y0_param = np.array([-1])
    y1_param = np.array([-1])
    
    # while sum(y0_param <= 0) != 0 and sum(y1_param <= 0) != 0:
    age = np.random.poisson(lam = 20, size = n_units) + 18
    edu = np.random.binomial(n = 1, p = 0.2, size = n_units)
    income = np.random.lognormal(mean = 1, sigma = 1, size = n_units)
    reward = np.random.binomial(n = 1, p = 0.3, size = n_units)
    duty = np.random.binomial(n = 1, p = 0.2, size = n_units)
    altruism = np.random.binomial(n = 1, p = 0.35, size = n_units)
    irrelevant = np.random.normal(loc = 10, scale = 10, size = [n_units, 10])
    # treatment = []
    # for i in range(n_units):
    #     try:
    #         treatment.append(np.random.binomial(n = 1, p = np.exp(edu[i] * 0.2 + income[i] * 0.1)/(1 + np.exp(edu[i] * 0.2 + income[i] * 0.1)), size = 1))
    #     except:
    #         print('dataset iteration', dataset_iteration, end = ' ')
    #         print('edu', edu[i], end = ' ')
    #         print('income', income[i], end = ' ')
    #         print('logit', np.exp(edu[i] * 0.2 + income[i] * 0.1))
    # treatment = np.array(treatment).reshape(n_units, )
    
    treatment = np.array([np.random.binomial(n = 1, p = np.exp(edu[i] * 0.2 + income[i] * 0.1)/(1 + np.exp(edu[i] * 0.2 + income[i] * 0.1)), size = 1) for i in range(n_units)])
    
    X = pd.DataFrame({
        'age' : age.reshape(n_units),
        'edu' : edu.reshape(n_units),
        'income' : income.reshape(n_units),
        'reward' : reward.reshape(n_units),
        'duty' : duty.reshape(n_units),
        'altruism' : altruism.reshape(n_units),
        # 'irrelevant' : irrelevant.reshape(n_units),
        'A' : treatment.reshape(n_units)
    }).merge(pd.DataFrame(irrelevant, columns = ['X' + str(i) for i in range(10)]), left_index=True, right_index=True)
    
    tip_no_tip_untreated_prob = np.array([np.exp(age[i] * 0.01 + income[i] * 0.03 + reward[i] * 0.19 + duty[i] * 0.06 + altruism[i] * 0.14)/(1 + np.exp(age[i] * 0.01 + income[i] * 0.03 + reward[i] * 0.19 + duty[i] * 0.06 + altruism[i] * 0.14)) for i in range(n_units)])
    tip_no_tip_treated_prob = np.array([np.exp(age[i] * 0.01 + income[i] * 0.03 + reward[i] * 0.19 + duty[i] * 0.06 + altruism[i] * 0.14 + 0.25)/(1 + np.exp(age[i] * 0.01 + income[i] * 0.03 + reward[i] * 0.19 + duty[i] * 0.06 + altruism[i] * 0.14 + 0.25)) for i in range(n_units)])
    
    tip_no_tip_untreated = np.array([np.random.binomial(n = 1,
                                     p = tip_no_tip_untreated_prob[i], 
                                     size = [1, n_obs_y]) for i in range(n_units)]).reshape([n_units, n_obs_y])
    tip_no_tip_treated = np.array([np.random.binomial(n = 1,
                                     p = tip_no_tip_treated_prob[i], 
                                     size = [1, n_obs_y]) for i in range(n_units)]).reshape([n_units, n_obs_y])
    
    tip_percent_treated_mean = np.exp(age * 0.01 + edu * 0.05 + reward * 0.13 + duty * -0.06 + altruism * 0.05)/(1 + np.exp(age * 0.01 + edu * 0.05 + reward * 0.13 + duty * -0.06 + altruism * 0.05))
    tip_percent_untreated_mean = np.exp(age * 0.01 + edu * 0.05 + reward * 0.13 + duty * -0.06 + altruism * 0.05 + 0.1)/(1 + np.exp(age * 0.01 + edu * 0.05 + reward * 0.13 + duty * -0.06 + altruism * 0.05 + 0.1))
    
    Y_untreated = np.array([np.random.beta(a = 10 * tip_percent_untreated_mean[i], 
                                           b = 10, 
                                           size = [1, n_obs_y]) for i in range(n_units)]).reshape([n_units, n_obs_y]) * tip_no_tip_untreated
    Y_treated = np.array([np.random.beta(a = 10 * tip_percent_treated_mean[i], 
                                           b = 10, 
                                           size = [1, n_obs_y]) for i in range(n_units)]).reshape([n_units, n_obs_y]) * tip_no_tip_treated
    
    y_obs = Y_treated * treatment + Y_untreated * (1 - treatment)
    y_unobs = Y_treated * (1 - treatment) + Y_untreated * (treatment)
    if 'dataset_' + str(seed) not in os.listdir('./experiments/tipping_sim_dgp/.'):
        os.mkdir('./experiments/tipping_sim_dgp/dataset_' + str(seed))
    X.to_csv('./experiments/tipping_sim_dgp/dataset_' + str(seed) + '/X.csv', index = False)
    pd.DataFrame(y_obs).to_csv('./experiments/tipping_sim_dgp/dataset_' + str(seed) + '/Y.csv', index = False)
    pd.DataFrame(y_unobs).to_csv('././experiments/tipping_sim_dgp/dataset_' + str(seed) + '/Y_unobs.csv', index = False)
    
    np.random.seed(999)
        
    # split into training and estimation datasets: 20% for training, 80% for estimation
    train_indexes = np.random.choice(range(n_units), size = int(0.2 * n_units), replace = False)
    est_indexes = list(set(range(n_units)) - set(train_indexes))
    X_train = X.iloc[train_indexes, :]
    X_est = X.iloc[est_indexes, :]
    y_train = y_obs[train_indexes, :]
    y_est = y_obs[est_indexes, :]
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
    ITE_df.to_csv('./experiments/tipping_sim_dgp/dataset_' + str(seed) + '/ITE.csv', index = False)
    

if __name__ == '__main__':
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(make_data, dataset_iterations_to_conduct)