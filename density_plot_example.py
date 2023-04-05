import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset_directory = './experiments/friedman_sim_dgp'
    Y = pd.read_csv(dataset_directory + '/dataset_2020/Y.csv')
    Y_unobs = pd.read_csv(dataset_directory + '/dataset_2020/Y_unobs.csv')
    X = pd.read_csv(dataset_directory + '/dataset_2020/X.csv')
    ITE_df = pd.read_csv(dataset_directory + '/dataset_2020/ITE.csv')
    
    Y_treated = Y.iloc[np.where(X['A'] == 1)[0][0], :].to_numpy().reshape(-1)
    print(Y_treated.shape)
    Y_untreated = Y_unobs.iloc[np.where(X['A'] == 1)[0][0], :].to_numpy().reshape(-1)
    print(Y_untreated.shape)
    A = np.hstack([np.repeat(1, Y_treated.shape[0]), np.repeat(0, Y_untreated.shape[0])])
    df = pd.DataFrame({
        'Y' : np.hstack([Y_treated, Y_untreated]),
        'Treatment' : A
    })
    
    sns.set_theme(style = 'white')
    sns.set(font_scale = 1.25)
    g = sns.kdeplot(data = df, x = 'Y', hue = 'Treatment')
    g.set_yticklabels([])
    g.set_xlabel(None)
    g.set_ylabel(None)
    plt.tight_layout()
    plt.show()
    plt.savefig(dataset_directory + '/density_plot.png', dpi = 300, transparent = True)
    plt.figure(0)
    ITE = ITE_df.iloc[np.where(X['A'] == 1)[0][0], :].to_numpy().reshape(-1)
    df = pd.DataFrame({
        'support' : np.linspace(0, 1, ITE.shape[0]),
        'ITE' : ITE
    })
    g = sns.lineplot(data = df, x = 'support', y = 'ITE')
    g.set_xlabel(None)
    g.set_ylabel(None)
    plt.tight_layout()
    plt.show()
    plt.savefig(dataset_directory + '/ite.png', dpi = 300, transparent = True)