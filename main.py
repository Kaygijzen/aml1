import numpy as np
import matplotlib.pyplot as plt

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from ConfigSpace.util import generate_grid

from sklearn.svm import SVR

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange

from assignment import SequentialModelBasedOptimization

import warnings
# warnings.filterwarnings("ignore")


class ModelTest():
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
        configurations = self.grid_cs(n_configs)
        n = len(configurations)

        perf = np.zeros(n)

        theta_inc_performance = 0

        for i, config in tqdm(enumerate(configurations), total=n):
            # print(f"grid search iteration {i+1}/{len(configurations)}")
            performance = self.optimizee(config)
            theta_inc_performance = max(theta_inc_performance, performance)
            # print(f"perf {theta_inc_performance}")
            # print('----------------------------')
            perf[i] = theta_inc_performance

        return perf
    
    def test_rand(self, n_configs):
        configs = self.sample_cs(n_configs)
        perf = np.zeros(n_configs)
        for i, config in tqdm(enumerate(configs), total=n_configs):
            perf[i] = max(perf[i-1], self.optimizee(config))
        return perf
    

    def test_smbo(self, n_iter, n_configs_init=8, n_configs_loop=2048):
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

        perf = np.zeros(n_iter)

        for idx in trange(n_iter):
            smbo.fit_model()
            config = smbo.select_configuration(sample_configs(n_configs_loop))
            performance = self.optimizee({hyp : val for hyp, val in zip(self.hyperparams, config)})
            smbo.update_runs((config, performance))
            perf[idx] = smbo.theta_inc_performance

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
        

def svr_test(datasets, n_configs):
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
        UniformFloatHyperparameter(
            name='alpha', lower=0.1, upper=.2, log=True, default_value=.15
        )
    ]

    from sklearn.linear_model import Lasso

    for dataset in datasets.items():
        test = ModelTest(
            Lasso,
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
        # 'iris': 61
    }
    # TODO : support classification datasets ? data.target needs to be converted from strings to ints
    # TODO : (optional) sommige datasets hebben pre-processing nodig van string bv naar one-hot encoding,
    #        misschien gewoon makkelijke datasets kiezen die dit niet hebben om het zo makkelijk mogelijk te maken

    svr_test(datasets, n_configs=16)
    # random_forest_test(datasets, n_configs=64)


if __name__ == '__main__':
    main()