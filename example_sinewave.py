import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import typing

from assignment import SequentialModelBasedOptimization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_configurations', type=int, default=16)
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/Downloads')
    parser.add_argument('--plot_resolution', type=int, default=128)
    parser.add_argument('--problem_size', type=int, default=1)

    return parser.parse_args()


def optimizee(x: float) -> typing.Union[float, np.array]:
    return np.sin(x[0])


def sample_configurations(n: int):
        x1 = np.random.uniform(X_MIN, X_MAX, (n, 1))
        return x1


def sample_initial_configurations(n: int) -> typing.List[typing.Tuple[np.array, float]]:
    configs = sample_configurations(n)
    return [(x, optimizee(x)) for x in configs]


def plot_surrogate(smbo, init_configs, plot_resolution, output_dir, n_init):
    # plot surrogate
    input_values = np.linspace(X_MIN, X_MAX, plot_resolution)
    
    plt.plot(input_values, [optimizee([val]) for val in input_values], color='red', label='ground truth')
    plt.plot(input_values, smbo.model.predict(input_values.reshape([-1, 1])), color='blue', label='surrogate')

    init_x_vals = [conf[0][0] for conf in init_configs[:n_init]]
    init_y_vals = [conf[1] for conf in init_configs[:n_init]]
    plt.scatter(init_x_vals, init_y_vals, color='green', label='initial configs')

    sampled_x_vals = [conf[0][0] for conf in init_configs[n_init:]]
    sampled_y_vals = [conf[1] for conf in init_configs[n_init:]]
    if len(sampled_x_vals) > 0:
        plt.scatter(sampled_x_vals, sampled_y_vals, color='m', label='sampled configs')

    # plt.axvline(next_config, label='next config', c='g')
    plt.xlabel('Input')
    plt.xlim([X_MIN, X_MAX])
    plt.legend()
    # plt.savefig('%s/result_n%d.png' % (output_dir, len(init_configs)))
    plt.show()


if __name__ == '__main__':
    args = parse_args()

    X_MIN = -np.pi * args.problem_size
    X_MAX = np.pi * args.problem_size

    np.random.seed(0)
    smbo = SequentialModelBasedOptimization()
    init_configs = sample_initial_configurations(args.initial_configurations)
    smbo.initialize(init_configs)
    smbo.fit_model()
    # plot_surrogate(smbo, init_configs, args.plot_resolution, args.output_directory, n_init=args.initial_configurations)

    input_values = np.linspace(X_MIN, X_MAX, 128).reshape(-1,1)
    for _ in range(5):
        smbo.fit_model()
        next_config = smbo.select_configuration(input_values)
        performance = optimizee(next_config)
        smbo.update_runs((next_config, performance))

    
    plot_surrogate(smbo, init_configs[:args.initial_configurations], args.plot_resolution, args.output_directory, n_init=args.initial_configurations)

    plot_surrogate(smbo, init_configs, args.plot_resolution, args.output_directory, n_init=args.initial_configurations)
