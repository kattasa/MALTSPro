import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool


def mise_ite(dataset_iteration, file_name):
    '''
    dataset_iteration : goes into seed
    '''
    
    folder = dataset_directory + '/dataset_' + str(2020 + 1000 * dataset_iteration)
    
    ITE_true_df = pd.read_csv(folder + '/ITE.csv').to_numpy()
    # try:
    ITE_est_df = pd.read_csv(folder + file_name, index_col='Unnamed: 0').to_numpy()
    
    # mise = ((np.abs(ITE_true_df - wass_tree_ITE_df)**2)).sum(axis = 1)/()
    mise = ((np.abs(ITE_true_df - ITE_est_df)**2)).sum(axis = 1)/((ITE_true_df**2).sum(axis = 1))
    print(file_name, mise.mean())
    return mise * 100
    # except:
    #     return np.repeat(a = np.nan, repeats = n_est_units) 

def parallel_plot(dataset_iteration):
    mise_list = []
    mise_df = pd.DataFrame({'Lin PS' : mise_ite(dataset_iteration, '/ps_ITE.csv'),
                            'df' : dataset_iteration})
    
    mise_df['FR'] = mise_ite(dataset_iteration, '/wass_ols_ITE.csv')
    mise_df['FR + Lin PS'] = mise_ite(dataset_iteration, '/dr_ITE.csv')
    mise_df['FR + RF PS'] = mise_ite(dataset_iteration, '/dr_rf_ITE.csv')
    mise_df['DrOut MALTS'] = mise_ite(dataset_iteration, '/maltspro_ITE.csv')
    mise_df['FT'] = mise_ite(dataset_iteration, '/wass_tree_ITE.csv')
    mise_df['FRF'] = mise_ite(dataset_iteration, '/wrf_ITE.csv')

    mise_list.append(mise_df)
    mise_df = pd.concat(mise_list)
    mise_df.to_csv(dataset_directory + '/mise.csv', index = False)
    
    return mise_df

def plot_mise(mise_df):
    mise_df = mise_df.rename(columns = {'MALTSPro' : 'DrOut MALTS'})
    include_methods = [
        'FR', 
        'FT',
        'FRF',
        'Lin PS',
        'FR + Lin PS',
        'FR + RF PS',
        'DrOut MALTS']
    mise_df = mise_df[include_methods].melt()
    # colors: "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"
    color_palette = {  
        'FR' : "#fd7f6f",
        'Lin PS' : "#b2e061",
        'FT' : "#ffee65",
        'FRF' : "#beb9db",
        'FR + Lin PS' : "#bd7ebe",
        'FR + RF PS' : "#ffb55a",
        'DrOut MALTS' : "#8bd3c7"
    }
    sns.set_theme(style = 'whitegrid')
    sns.set(rc = {'figure.figsize':(12,10)})
    sns.set(font_scale = 1.5)
    sns.boxplot(data = mise_df, x = 'variable', y = 'value', showfliers = False, palette=[color_palette[i] for i in include_methods])
    plt.ylabel('Relative Error (%)')
    plt.xlabel(None)
    # plt.xticks(rotation = 45)
    plt.tight_layout()
    
    plt.savefig(dataset_directory + '/plot.png', dpi = 300, transparent = True)
    
if __name__ == '__main__':
    dataset_directory = './experiments/quadratic_sim_dgp'
    n_est_units = 800
    try:
        mise_df = pd.read_csv(dataset_directory + '/mise.csv')
        plot_mise(mise_df)
    except:
        dataset_iterations_to_conduct = range(0, 100)
        with Pool(processes = 25) as pool:
            dfs = pool.map(parallel_plot,
                    dataset_iterations_to_conduct)
        mise_df = pd.concat(dfs, axis = 0)
        mise_df.to_csv(dataset_directory + '/mise.csv', index = False)
        plot_mise(mise_df)