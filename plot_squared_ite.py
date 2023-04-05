import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def true_ite(dataset_directory, dataset_iteration):
    folder = dataset_directory + '/dataset_' + str(2020 + 1000 * dataset_iteration)
    ITE_true_df = pd.read_csv(folder + '/ITE.csv').to_numpy()
    squared_ite = ((ITE_true_df**2).sum(axis = 1))
    return squared_ite
    
    
if __name__ == '__main__':
    n_est_units = 800
    df_list = []
    for dataset_iteration in range(100):
        print(dataset_iteration)
        squared_ite_df = pd.DataFrame({'Linear' : true_ite('./experiments/friedman_normal_sim_dgp', dataset_iteration),
                                'df' : dataset_iteration})
        
        squared_ite_df['Mixture'] = true_ite('./experiments/tipping_sim_dgp', dataset_iteration)
        squared_ite_df['Friedman Normal'] = true_ite('./experiments/friedman_normal_sim_dgp', dataset_iteration)
        squared_ite_df['Friedman Beta'] = true_ite('./experiments/friedman_sim_dgp', dataset_iteration)
        
        df_list.append(squared_ite_df)
    squared_ite_df = pd.concat(df_list)
    # mise_df.to_csv(dataset_directory + '/mise.csv', index = False)
    
    squared_ite_df.boxplot(column = [
        'Linear', 
                                           'Mixture', 
                                           'Friedman Normal',
                                        'Friedman Beta'
                                           ], showfliers = False)
    plt.ylabel('Squared True ITE')
    plt.savefig('./true_squared_ite_plot.png')