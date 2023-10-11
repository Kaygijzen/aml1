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
    def __init__(self) -> None:
        self.cs = ConfigSpace.ConfigurationSpace('boston.cs', 1)

        C = ConfigSpace.UniformFloatHyperparameter(
            name='C', lower=0.03125, upper=32768, log=True, default_value=1.0)
        gamma = ConfigSpace.UniformFloatHyperparameter(
            name='gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
        self.cs.add_hyperparameters([C, gamma])

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
    

    def test_smbo():
        pass
        # return perf array

    def test_grid(self, n_configurations):
        grid = self.grid_cs(n_configurations)


        for config in grid:
            print(config["C"], config["gamma"])
            break
        # return perf array

    def test_random():
        pass
        # return perf array


    def run():
        # run all test
        # and plot perf's
        pass



def svm_boston_smbo():
    np.random.seed(1)

    data = sklearn.datasets.fetch_openml(data_id=43465)
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        data.data, data.target, test_size=0.33, random_state=1)

    def optimizee(gamma, C):
        clf = sklearn.svm.SVR()
        clf.set_params(kernel='rbf', gamma=gamma, C=C)
        clf.fit(X_train, y_train)
        return clf.score(X_valid, y_valid)#sklearn.metrics.accuracy_score(y_valid, clf.predict(X_valid))


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
        return [((gamma, C), optimizee(gamma, C)) for gamma, C in configs]


    smbo = SequentialModelBasedOptimization()
    smbo.initialize(sample_initial_configurations(16))

    print(smbo.theta_inc_performance)
    print(smbo.theta_inc_performance)
    print('starting loop')

    n = 256
    perf = np.zeros(n)

    for idx in range(n):
        print('iteration %d/64' % idx)
        smbo.fit_model()
        theta_new = smbo.select_configuration(sample_configurations(1024))
        performance = optimizee(theta_new[0], theta_new[1])
        # plt.plot(performance)
        # plt.show()
        print(performance)
        smbo.update_runs((theta_new, performance))
        print(smbo.theta_inc_performance)
        print('----------------------------')
        perf[idx] = smbo.theta_inc_performance

    return perf

def svm_boston_grid():
    pass

def svm_boston_random():
    pass

def main():
    perf = svm_boston_smbo()

    # # print(perf)

    plt.plot(perf)
    plt.show()

    # boston = BostonTest()
    # grid = boston.grid(16)
    # print(len(grid))

    # boston.test_grid(10)


if __name__ == '__main__':
  main()