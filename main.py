import ConfigSpace
from ConfigSpace.util import generate_grid
import numpy as np
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.svm

from tqdm.notebook import trange

from assignment import SequentialModelBasedOptimization

import typing

import warnings
warnings.filterwarnings("ignore")


class BostonTest():
    data = sklearn.datasets.fetch_openml(data_id=43465)
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        data.data, data.target, test_size=0.33, random_state=1)
    
    def __init__(self) -> None:
        self.cs = ConfigSpace.ConfigurationSpace('boston.cs', 1)

        C = ConfigSpace.UniformFloatHyperparameter(
            name='C', lower=0.03125, upper=32768, log=True, default_value=1.0)
        gamma = ConfigSpace.UniformFloatHyperparameter(
            name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
        self.cs.add_hyperparameters([C, gamma])

        self.theta_inc_performance = 0
    
    def optimizee(self, gamma, C):
        clf = sklearn.svm.SVR()
        clf.set_params(kernel='rbf', gamma=gamma, C=C)
        clf.fit(self.X_train, self.y_train)
        return clf.score(self.X_valid, self.y_valid)

    def grid_cs(self, n_configurations):
        n = int(round(np.sqrt(n_configurations)))
        return generate_grid(
            configuration_space=self.cs, 
            num_steps_dict={"C": n, "gamma": n}
        )

    def sample_cs(self, n_configurations):
        return np.array([(
            configuration['gamma'],
            configuration['C'])
            for configuration in self.cs.sample_configuration(n_configurations)
        ])
    
    def test_grid(self, n_configurations):
        configurations = self.grid_cs(n_configurations)
        n = len(configurations)
        perf = np.zeros(n)

        theta_inc_performance = 0

        for i, theta_new in enumerate(configurations):
            print(f"grid search iteration {i+1}/{len(configurations)}")
            performance = self.optimizee(theta_new["gamma"], theta_new["C"])
            theta_inc_performance = max(theta_inc_performance, performance)
            print(f"perf {theta_inc_performance}")
            print('----------------------------')
            perf[i] = theta_inc_performance

        return perf

    def test_random(self, n_configurations):
        svr = sklearn.svm.SVR()
        distributions = self.cs.get_hyperparameters_dict()
        clf = sklearn.model_selection.RandomizedSearchCV(svr, distributions, n_iter=n_configurations)
        search = clf.fit(self.X_train, self.y_train)
        return search.score(self.X_valid, self.y_valid)

    def test_smbo(self, n_configurations):
        np.random.seed(0)

        data = sklearn.datasets.fetch_openml(data_id=43465)
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
            data.data, data.target, test_size=0.33, random_state=1)

        def sample_configurations(n_configurations):
            # function uses the ConfigSpace package, as developed at Freiburg University.
            # most of this functionality can also be achieved by the scipy package
            # same hyperparameter configuration as in scikit-learn
            cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVR', 1)

            C = ConfigSpace.UniformFloatHyperparameter(
                name='C', lower=0.03125, upper=32768, log=True, default_value=1.0)
            gamma = ConfigSpace.UniformFloatHyperparameter(
                name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
            cs.add_hyperparameters([C, gamma])

            return np.array([(configuration['gamma'],
                                configuration['C'])
                            for configuration in cs.sample_configuration(n_configurations)])

        def sample_initial_configurations(n: int) -> typing.List[typing.Tuple[np.array, float]]:
            configs = sample_configurations(n)
            return [((gamma, C), self.optimizee(gamma, C)) for gamma, C in configs]


        smbo = SequentialModelBasedOptimization()
        smbo.initialize(sample_initial_configurations(16))

        n = 64
        perf = np.zeros(n)

        for idx in range(n):
            print('iteration %d/64' % idx)
            smbo.fit_model()
            theta_new = smbo.select_configuration(sample_configurations(1024))
            performance = self.optimizee(theta_new[0], theta_new[1])
            # plt.plot(performance)
            # plt.show()
            print(performance)
            smbo.update_runs((theta_new, performance))
            print(smbo.theta_inc_performance)
            print('----------------------------')
            perf[idx] = smbo.theta_inc_performance

        return perf

    def run(self, n):
        # Grid search
        grid_search_performance = self.test_grid(n)

        # Random search
        random_search_performance = self.test_random(n)
        
        #SMBO
        smbo_performance = self.test_smbo(n)

        # Plotting performance
        plt.plot(grid_search_performance, label="Grid Search")
        plt.axhline(y = random_search_performance, color="r", label="Random Search") 
        plt.plot(smbo_performance, label="SMBO")
        plt.legend()
        plt.show()


def main():
    BostonTest().run(64)


if __name__ == '__main__':
  main()