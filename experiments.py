import numpy as np
import matplotlib.pyplot as plt

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.util import generate_grid

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange
import pickle
from glob import glob

from assignment import SequentialModelBasedOptimization

import warnings
warnings.filterwarnings("ignore")


class HPOptimization():
    # Class that runs hyperparameter optimization 
    # using SMBO, grid search and random search, on given datasets
    def __init__(self, model, dataset, hyperparameter_space, random_state=0):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.model = model
        self.dataset_name,  self.dataset_id = dataset
        data = fetch_openml(data_id=self.dataset_id)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            data.data, data.target, test_size=0.33, random_state=self.random_state
        )

        self.cs = ConfigurationSpace(self.dataset_name, self.random_state)
        self.cs.add_hyperparameters(hyperparameter_space)
        self.hyperparams = list(self.cs.keys())

        self.grid_perf = self.smbo_perf = self.rand_perf = None

    def optimizee(self, config):
        model = self.model()
        config = {key: val if round(val) != val else int(val) for key, val in config.items()}
        model.set_params(**config)
        model.fit(self.X_train, self.y_train)
        return model.score(self.X_valid, self.y_valid)
    
    def grid_cs(self, n_configs):
        # `n` number of steps per dimension i.e. hyperparameter
        n = int(round(n_configs**(1/len(self.cs))))
        return generate_grid(
            configuration_space=self.cs, 
            num_steps_dict={hyp : n for hyp in self.hyperparams}
        )
    
    def sample_cs(self, n_configs):
        return self.cs.sample_configuration(n_configs)
    
    def perform_grid_search(self, n_configs):
        # Perform grid search over hyperparameter space
        self.configs_grid = self.grid_cs(n_configs)
        perf = np.zeros(n_configs)
        for i, config in tqdm(enumerate(self.configs_grid), total=n_configs):
            perf[i] = max(perf[i-1], self.optimizee(config))
        return perf
    
    def perform_random_search(self, n_configs):
        # Perform random search over hyperparameter space
        self.configs_rand = self.sample_cs(n_configs)
        perf = np.zeros(n_configs)
        for i, config in tqdm(enumerate(self.configs_rand), total=n_configs):
            perf[i] = max(perf[i-1], self.optimizee(config))
        return perf
    
    def perform_smbo(self, n_iter, n_configs_init=3, n_configs_loop=128):
        def sample_configs(n_configs):
            return np.array([
                [value for _, value in config.items()]
                for config in self.sample_cs(n_configs)
            ])

        def sample_init_configs(configs):
            return [(
                np.array([value for _, value in config.items()]),
                self.optimizee(config)
            ) for config in configs]        
        
        smbo = SequentialModelBasedOptimization()
        smbo.initialize(sample_init_configs(self.sample_cs(n_configs_init)))

        perf = np.zeros(n_iter).astype(float)
        perf[:n_configs_init] = np.nan

        for idx in trange(n_iter - n_configs_init):
            smbo.fit_model()
            config = smbo.select_configuration(sample_configs(n_configs_loop))
            performance = self.optimizee({hyp : val for hyp, val in zip(self.hyperparams, config)})
            smbo.update_runs((config, performance))
            perf[idx+n_configs_init] = smbo.theta_inc_performance

        return perf, smbo
    
    def run(self, n_configs, smbo_n_init_configs, smbo_n_samples):
        print('grid search')
        self.grid_perf = self.perform_grid_search(n_configs)
        print('random search')
        self.rand_perf = self.perform_random_search(n_configs)
        print('smbo search')
        self.smbo_perf, self.smbo = self.perform_smbo(n_configs, smbo_n_init_configs, smbo_n_samples)
       
    def show(self):
        plt.plot(self.grid_perf, label="Grid Search")
        plt.plot(self.rand_perf, label="Random Search")
        plt.plot(self.smbo_perf, label="SMBO")
        plt.legend()
        plt.show()
        

def svr_grid_exp(dataset_id, n_configs=36, smbo_n_init_configs=8, smbo_n_samples=64):
    hyperparameter_space = [
        UniformFloatHyperparameter(
            name='C', lower=0.03125, upper=32768, log=True, default_value=1.0
        ),
        UniformFloatHyperparameter(
            name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1
        )
    ]

    from sklearn.svm import SVR

    test = HPOptimization(
        SVR,
        dataset_id,
        hyperparameter_space,
        random_state=1
    )

    test.run(n_configs, smbo_n_init_configs, smbo_n_samples)

    # Plot the performance on the test set, over no. of optimization iterations
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    x = np.arange(n_configs)+1
    ax[0].plot(x, test.grid_perf, label="Grid Search")
    ax[0].plot(x, test.rand_perf, label="Random Search")
    ax[0].plot(x, test.smbo_perf, label="SMBO")
    ax[0].set_xlabel('no. iterations')
    ax[0].set_ylabel('test score')
    ax[0].legend(frameon=False)

    C_vals, gamma_vals = zip(*[(c['C'],c['gamma']) for c in test.configs_grid])
    grid_best = test.configs_grid[test.grid_perf.argmax()]
    ax[1].loglog(C_vals, gamma_vals, 'C0.', ms=10)
    ax[1].plot(grid_best['C'], grid_best['gamma'], 'C0.', ms=20)

    C_vals, gamma_vals = zip(*[(c['C'],c['gamma']) for c in test.configs_rand])
    rand_best = test.configs_rand[test.rand_perf.argmax()]
    ax[1].plot(C_vals, gamma_vals, 'C1.', ms=10, alpha=.2)
    ax[1].plot(rand_best['C'], rand_best['gamma'], 'C1.', ms=20)
    
    sbmo_configs = np.vstack([c for c, _ in test.smbo.capital_r])[4:]
    ax[1].plot(*sbmo_configs.T, 'C2.', ms=10)
    ax[1].plot(*sbmo_configs[np.argmax(test.smbo_perf)].T, 'C2.', ms=20) 
    ax[1].set_xlabel('C')
    ax[1].set_ylabel('gamma')

    fig.tight_layout()
    fig.savefig('./svr_grid.pdf')


def svr_experiment(datasets, n_configs, smbo_n_init_configs, smbo_n_samples):
    # Hyperparameter optimization of a SVM Regressor
    hyperparameter_space = [
        UniformFloatHyperparameter(
            name='C', lower=1e-2, upper=1e3, log=True, default_value=1.0
        ),
        UniformFloatHyperparameter(
            name='gamma', lower=1e-04, upper=10, log=True, default_value=0.1
        )
    ]

    from sklearn.svm import SVR

    expriments = [
        HPOptimization(SVR,dataset,hyperparameter_space,random_state=0) 
        for dataset in datasets.items()
    ]
    for exp in expriments : exp.run(n_configs, smbo_n_init_configs, smbo_n_samples)

    return expriments


def random_forest_experiment(datasets, n_configs, smbo_n_init_configs, smbo_n_samples):
    # Hyperparameter optimization of a Random Forest Regressor
    hyperparameter_space = [
        UniformIntegerHyperparameter(
            name='n_estimators', lower=8, upper=512, log=True, default_value=32
        ),
        UniformIntegerHyperparameter(
            name='max_depth', lower=2, upper=64, log=True, default_value=8
        ),
        UniformIntegerHyperparameter(
            name='min_samples_split', lower=2, upper=32, log=True, default_value=8
        ),
        UniformIntegerHyperparameter(
            name='min_samples_leaf', lower=2, upper=32, log=True, default_value=8
        ),
        UniformFloatHyperparameter(
            name='max_features', lower=.1, upper=.9, log=True, default_value=.9
        )
    ]

    from sklearn.ensemble import RandomForestRegressor

    # Run optimization on all datasets
    expriments = [
        HPOptimization(RandomForestRegressor, dataset, hyperparameter_space, random_state=0) 
        for dataset in datasets.items()
    ]
    for exp in expriments : exp.run(n_configs, smbo_n_init_configs, smbo_n_samples)

    return expriments


def plot_expr(results, datasets, ax):
    # Function that plots the experiment results
    labels = ['grid', 'random', 'SMBO']

    x = np.arange(len(datasets))
    x_sub = np.array([-.15, 0, .15])
    for i, res in enumerate(results):
        ax.bar(x + x_sub[i], res, width=.1, label=labels[i])
    ax.set_xticks(x)
    ax.set_xticklabels([key for key, _ in datasets.items()])


def plot_config():
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', family='serif', serif="cmr10", size=16)
    plt.rc('mathtext', fontset='cm', rm='serif')
    plt.rc('axes', unicode_minus=False)
    plt.rcParams['axes.formatter.use_mathtext'] = True


def main():
    # datasets from https://www.openml.org/search?type=data&status=active
    datasets = {
        'fat' : 560,
        'boston' : 43465,
        'wine' : 43994,
    }

    # If hyperparameter optimization has been run before, we won't have to do this again.
    pickle_files = glob('*.pkl')
    if 'svr_results.pkl' not in pickle_files:
        svr_results = svr_experiment(datasets, n_configs=16, smbo_n_init_configs=3, smbo_n_samples=128)
        with open('svr_results.pkl', 'wb') as handle:
            pickle.dump(svr_results, handle)
    else:
        with open('svr_results.pkl', 'rb') as handle:
            svr_results = pickle.load(handle)

    if 'rf_results.pkl' not in pickle_files:
        rf_results = random_forest_experiment(datasets, n_configs=243, smbo_n_init_configs=16, smbo_n_samples=128) # (3^5) = 243 we have 5 hyperparameters
        with open('rf_results.pkl', 'wb') as handle:
            pickle.dump(rf_results, handle)
    else:
        with open('rf_results.pkl', 'rb') as handle:
            rf_results = pickle.load(handle)


    plot_config()

    # The SVM Regressor experiment on the boston dataset
    svr_grid_exp(('boston', 43465))


    # Plot bar charts of final test score of the SVM and RF Regressor on the 3 datasets
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5,5))

    svr_final_perf = zip(*[(exp.grid_perf[-1], exp.rand_perf[-1], exp.smbo_perf[-1]) for exp in svr_results])
    plot_expr(svr_final_perf, datasets, ax[0])
    ax[0].set_title('SVM')

    rf_final_perf = zip(*[(exp.grid_perf[-1], exp.rand_perf[-1], exp.smbo_perf[-1]) for exp in rf_results])
    plot_expr(rf_final_perf, datasets, ax[1])
    ax[1].set_title('Random Forest')

    ax[0].set_ylabel('final test score')

    fig.tight_layout()
    fig.savefig('./final_perf.pdf')


    # Plot bar charts of average test score of the SVM and RF Regressor on the 3 datasets
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(5,5))

    svr_final_perf = zip(*[(exp.grid_perf[3:].mean(), exp.rand_perf[3:].mean(), exp.smbo_perf[3:].mean()) for exp in svr_results])

    plot_expr(svr_final_perf, datasets, ax[0])
    ax[0].set_title('SVM')

    rf_final_perf = zip(*[(exp.grid_perf[16:].mean(), exp.rand_perf[16:].mean(), exp.smbo_perf[16:].mean()) for exp in rf_results])
    plot_expr(rf_final_perf, datasets, ax[1])
    ax[1].set_title('Random Forest')

    ax[0].set_ylabel('average test score')

    fig.tight_layout()
    fig.savefig('./avg_perf.pdf')

    plt.show()


if __name__ == '__main__':
    main()