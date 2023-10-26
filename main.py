import numpy as np
import matplotlib.pyplot as plt

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.util import generate_grid

from sklearn.svm import SVR

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange

from assignment import SequentialModelBasedOptimization

import warnings
warnings.filterwarnings("ignore")


class ModelTest():
    def __init__(self, model, dataset, hyperparameter_space, random_state=1):
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
    
    def test_grid(self, n_configs):
        self.configs_grid = self.grid_cs(n_configs)
        perf = np.zeros(n_configs)
        for i, config in tqdm(enumerate(self.configs_grid), total=n_configs):
            perf[i] = max(perf[i-1], self.optimizee(config))
        return perf
    
    def test_rand(self, n_configs):
        self.configs_rand = self.sample_cs(n_configs)
        perf = np.zeros(n_configs)
        for i, config in tqdm(enumerate(self.configs_rand), total=n_configs):
            perf[i] = max(perf[i-1], self.optimizee(config))
        return perf
    

    def test_smbo(self, n_iter, n_configs_init=3, n_configs_loop=128):
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
    
    def test_smbo_normalized(self, n_iter, n_configs_init=3, n_configs_loop=128):
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

        sample = self.sample_cs(n_configs_init)

        smbo.initialize(sample_init_configs(sample))

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
        self.grid_perf = self.test_grid(n_configs)
        print('random search')
        self.rand_perf = self.test_rand(n_configs)
        print('smbo search')
        self.smbo_perf, self.smbo = self.test_smbo(n_configs, smbo_n_init_configs, smbo_n_samples)
       
    def show(self):
        plt.plot(self.grid_perf, label="Grid Search")
        plt.plot(self.rand_perf, label="Random Search")
        plt.plot(self.smbo_perf, label="SMBO")
        plt.legend()
        plt.show()

    
        

def svr_test(datasets, n_configs, smbo_n_init_configs=8, smbo_n_samples=128):
    hyperparameter_space = [
        UniformFloatHyperparameter(
            name='C', lower=0.03125, upper=32768, log=True, default_value=1.0
        ),
        UniformFloatHyperparameter(
            name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1
        )
    ]

    for dataset in datasets.items():
        test = ModelTest(
            SVR,
            dataset,
            hyperparameter_space,
            random_state=1
        )

        test.run(n_configs, smbo_n_init_configs, smbo_n_samples)

        fig, ax = plt.subplots(1, 2, figsize=(10,5))

        ax[0].plot(test.grid_perf, label="Grid Search")
        ax[0].plot(test.rand_perf, label="Random Search")
        ax[0].plot(test.smbo_perf, label="SMBO")
        ax[0].set_xlabel('iterations')
        ax[0].set_ylabel('score')
        ax[0].legend(frameon=False)
    
        C_vals, gamma_vals = zip(*[(c['C'],c['gamma']) for c in test.configs_grid])
        ax[1].loglog(C_vals, gamma_vals, 'C0.', ms=10)
        grid_best = test.configs_grid[test.grid_perf.argmax()]
        ax[1].plot(grid_best['C'], grid_best['gamma'], 'C0.', ms=20)

        C_vals, gamma_vals = zip(*[(c['C'],c['gamma']) for c in test.configs_rand])
        ax[1].plot(C_vals, gamma_vals, 'C1.', ms=10, alpha=.2)
        rand_best = test.configs_rand[test.rand_perf.argmax()]
        ax[1].plot(rand_best['C'], rand_best['gamma'], 'C1.', ms=20)
        
        # TODO smbo_n_itit configs toevoegen
        sbmo_configs = np.vstack([c for c, _ in test.smbo.capital_r])[4:]
        ax[1].plot(*sbmo_configs.T, 'C2.', ms=10)
        ax[1].plot(*sbmo_configs[np.argmax(test.smbo_perf)].T, 'C2.', ms=20) 
        ax[1].set_xlabel('C')
        ax[1].set_ylabel('gamma')

        fig.tight_layout()
        plt.show()


def random_forest_test(datasets, n_configs, smbo_n_init_configs=8, smbo_n_samples=128):
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

    for dataset in datasets.items():
        test = ModelTest(
            RandomForestRegressor,
            dataset,
            hyperparameter_space,
            random_state=0
        )

        test.run(n_configs, smbo_n_init_configs, smbo_n_samples)
        test.show()



def main():
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.rc('font', family='serif', serif="cmr10", size=16)
    plt.rc('mathtext', fontset='cm', rm='serif')
    plt.rc('axes', unicode_minus=False)

    plt.rcParams['axes.formatter.use_mathtext'] = True

    # datasets : https://www.openml.org/search?type=data&status=active
    datasets = {
        'boston-housing' : 43465,
        'quake': 209,
        'wine_quality': 43994
    }

    svr_test(datasets, n_configs=36, smbo_n_init_configs=8, smbo_n_samples=256)
    random_forest_test(datasets, n_configs=36, smbo_n_init_configs=8, smbo_n_samples=256)


if __name__ == '__main__':
    main()