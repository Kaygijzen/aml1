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
        configs = self.grid_cs(n_configs)
        perf = np.zeros(n_configs)
        for i, config in tqdm(enumerate(configs), total=n_configs):
            perf[i] = max(perf[i-1], self.optimizee(config))
        return perf
    
    def test_rand(self, n_configs):
        configs = self.sample_cs(n_configs)
        perf = np.zeros(n_configs)
        for i, config in tqdm(enumerate(configs), total=n_configs):
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

        return perf

    def run(self, n_configs):
        print('grid search')
        self.grid_perf = self.test_grid(n_configs)
        print('random search')
        self.rand_perf = self.test_rand(n_configs)
        print('smbo search')
        self.smbo_perf = self.test_smbo(n_configs)
       
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

        test.run(n_configs=n_configs)
        test.show()


def random_forest_test(datasets, n_configs):
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

        test.run(n_configs=n_configs)
        test.show()


def main():
    # datasets : https://www.openml.org/search?type=data&status=active
    datasets = {
        'boston-housing' : 43465,
        'quake': 209,
        'wine_quality': 43994
    }

    # TODO : (optional) sommige datasets hebben pre-processing nodig van string bv naar one-hot encoding,
    #        misschien gewoon makkelijke datasets kiezen die dit niet hebben om het zo makkelijk mogelijk te maken

    svr_test(datasets, n_configs=25, smbo_n_init_configs=8, smbo_n_samples=128)
    # random_forest_test(datasets, n_configs=243)


if __name__ == '__main__':
    main()