import pandas as pd
import numpy as np
from pymaltspro2 import linbo_ITE
import pickle as pkl
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression

class ols_quantile():
    def __init__(self, qtl_grid, treatment_var):
        self.qtl_grid = np.array(qtl_grid)
        self.treatment_var = treatment_var
    
    def fit(self, X_train, Y_train):
        A_train = X_train[self.treatment_var].values
        X_train = X_train.drop(self.treatment_var, axis = 1)
        
        control = np.where(A_train == 0)[0]
        X_control = X_train.iloc[control, :]
        Y_control = Y_train[control, :]
        
        treated = np.where(A_train == 1)[0]
        X_treated = X_train.iloc[treated, :]
        Y_treated = Y_train[treated, :]
        
        self.OLS_control_list = []
        self.OLS_treated_list = []
        for q in self.qtl_grid:
            OLS_control = LinearRegression()
            OLS_control.fit(X_control, Y_control[:, q])
            self.OLS_control_list.append(OLS_control)
            
            OLS_treated = LinearRegression()
            OLS_treated.fit(X_treated, Y_treated[:, q])
            self.OLS_treated_list.append(OLS_treated)
        self.OLS_control_list = np.array(self.OLS_control_list)
        self.OLS_treated_list = np.array(self.OLS_treated_list)
    
    def predict_control(self, X_est):
        X_est = X_est.drop(self.treatment_var, axis = 1)
        Y_predict_control = np.ones(shape = [X_est.shape[0], self.qtl_grid.shape[0]])
        print(self.OLS_control_list.shape)
        for q in self.qtl_grid:
            Y_predict_control[:, q] = self.OLS_control_list[q].predict(X_est).reshape(Y_predict_control[:, q].shape)
        # Y_predict_control = np.apply_along_axis(func1d = lambda model: model.predict(X_est), axis = 0, arr = self.OLS_control_list)
        Y_predict_control = Y_predict_control.reshape([X_est.shape[0], self.qtl_grid.shape[0]])
        # for i in range(X_est.shape[0]):
        #     Y_control_i = []
            
        #     for q in self.qtl_grid:
        #         Y_control_i.append(self.OLS_control_list[q].predict(X_est.iloc[i:(i+1), :]))
        #     Y_control_i = np.array(Y_control_i)
        #     Y_predict_control.append(Y_control_i)
        Y_predict_control = np.array(Y_predict_control)
        return Y_predict_control
            
    def predict_treated(self, X_est):
        X_est = X_est.drop(self.treatment_var, axis = 1)
        Y_predict_treated = np.ones(shape = [X_est.shape[0], self.qtl_grid.shape[0]])
        print(self.OLS_treated_list.shape)
        for q in self.qtl_grid:
            Y_predict_treated[:, q] = self.OLS_treated_list[q].predict(X_est).reshape(Y_predict_treated[:, q].shape)
        # Y_predict_treated = np.apply_along_axis(func1d = lambda model: model.predict(X_est), axis = 0, arr = self.OLS_treated_list)
        Y_predict_treated = Y_predict_treated.reshape([X_est.shape[0], self.qtl_grid.shape[0]])
        return Y_predict_treated
        
        
def ols_meta_predict(X_valid, wass_ols, treatment_var):
    y_pred = []
    # for i in range(X_valid.shape[0]):
    #     if X_valid.loc[i, treatment_var] == 1:
    #         # impute treated cf
    #         y_pred.append(wass_ols.predict_control(X_valid.iloc[i:(i+1), :])[0, :])
    #     else:
    #         # impute control cf
    #         y_pred.append(wass_ols.predict_treated(X_valid.iloc[i:(i+1), :])[0, :])
    A = X_valid[treatment_var].values
    A_repeat = np.array([np.repeat(A_i, repeats = wass_ols.qtl_grid.shape[0], axis = 0) for A_i in A])
    X_valid[treatment_var] = 0
    y_pred_control = wass_ols.predict_control(X_valid)
    X_valid[treatment_var] = 1
    y_pred_treated = wass_ols.predict_treated(X_valid)
    print(y_pred_treated.shape, y_pred_control.shape)
    y_pred = (1 - A_repeat) * y_pred_treated + A_repeat * y_pred_control
    return y_pred

### Wasserstein regression experiments
def wass_ols_parallel(dataset_iteration):
    print(dataset_iteration, end = ' ')
    seed = 2020 + 1000 * dataset_iteration

    # read dataset
    maltspro_df = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/X.csv')
    y = pd.read_csv(dataset_directory + '/dataset_' + str(seed) + '/Y.csv').to_numpy()

    # turn into quantile functions
    #   1. find #quantiles
    n_samples_min = np.apply_along_axis(
                arr = y,
                axis = 1,
                func1d = lambda x: x[x == x].shape[0]).min()
    #   2. estimate quantile function at grid
    qtls = range(n_samples_min)
    y_qtl = np.apply_along_axis(
                    arr = y,
                    axis = 1,
                    func1d = lambda x: np.quantile(
                        a = x, 
                        q = np.linspace(start = 0, stop = 1, num = n_samples_min)
                        )
                    ).reshape(y.shape)

    n_units = maltspro_df.shape[0]
    # make training, estimation split

    print('training')
    # split into training and estimation datasets: 20% for training, 80% for estimation
    train_indexes = np.random.choice(range(n_units), size = int(0.2 * n_units), replace = False)
    est_indexes = list(set(range(n_units)) - set(train_indexes))

    X_train = maltspro_df.iloc[train_indexes, :].reset_index().drop('index', axis = 1)
    X_est = maltspro_df.iloc[est_indexes, :].reset_index().drop('index', axis = 1)

    y_train = y_qtl[train_indexes, :]
    y_est = y_qtl[est_indexes, :]

    # train control and treated regressions
    wass_ols = ols_quantile(qtl_grid=qtls, treatment_var='A')
    wass_ols.fit(X_train, y_train)


    # save OLS models
    treated_file_name = open(dataset_directory + '/dataset_' + str(seed) + '/wass_ols.pkl', 'wb')
    pkl.dump(wass_ols, treated_file_name)

    print('imputing counterfactuals')
    # impute counterfactuals
    y_wass_ols_bary = ols_meta_predict(X_est, wass_ols, treatment_var = 'A').reshape(y_est.shape)

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

    print('estimating treatment effects')
    # estimate the ITE with imputed counterfactuals
    ITE_wass_ols = []
    for i in range(len(est_indexes)):
        ITE_wass_ols.append(
            linbo_ITE(
                y_obs = y_est[i, :],
                y_cf = y_wass_ols_bary[i, :],
                observed_treatment = X_est['A'].values[i],
                reference_distribution = np.vstack([np.linspace(0, 1, n_samples_min),
                                                    np.linspace(0, 1, n_samples_min)]),
                y_obs_qtl_id = False, 
                y_cf_qtl_id = True
            )[1, :]
        )
    ITE_wass_ols = np.array(ITE_wass_ols)
    ATE_wass_ols = ITE_wass_ols.mean(axis = 1)
    print(dataset_iteration, ': WassOLS ATE min =', ATE_wass_ols.min(), ' ATE max = ', ATE_wass_ols.max())

    print('saving data')
    wass_ols_ITE_df = pd.DataFrame(ITE_wass_ols, columns = np.linspace(0, 1, ITE_wass_ols.shape[1]))
    wass_ols_ITE_df.to_csv(dataset_directory + '/dataset_' + str(seed) + '/wass_ols_ITE.csv')

if __name__ == '__main__':
    dataset_directory = './experiments/friedman_sim_dgp'
    # opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    # for opt, arg in opts:
    #   if opt == '-f':
    #     dataset_directory = arg 
    dataset_iterations_to_conduct = range(0, 100)
    with Pool(processes = 25) as pool:
        pool.map(wass_ols_parallel,
                    dataset_iterations_to_conduct)