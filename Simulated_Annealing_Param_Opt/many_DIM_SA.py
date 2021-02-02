import sys, os
sys.path.append(os.getcwd())    # need this for importing to work nicely

import pickle
import time
from Utils import multiple_runs_with_different_seed
import numpy as np
import pandas as pd
from SimulatedAnealing import SimulatedAnnealing
from rana import rana_func, Rosenbrock
from concurrent.futures import ProcessPoolExecutor


def run(config,  Class=SimulatedAnnealing, n_runs=20):
    dataframe = pd.DataFrame()
    results = multiple_runs_with_different_seed(Class=Class, class_argument=config, n_iterations=n_runs)
    mean_performance = np.mean(results[:, 1])
    best_performance = np.min(results[:, 1])
    std_perormance = np.std(results[:, 1])
    average_runtime = np.mean(results[:, 2])

    mean_performance_final = np.mean(results[:, 0])
    best_performance_final = np.min(results[:, 0])
    std_perormance_final = np.std(results[:, 0])

    config_result = config.copy()
    config_result["raw_results"] = results[:, 1]
    config_result["mean_performance"] = mean_performance
    config_result["best_performance"] = best_performance
    config_result["std_perormance"] = std_perormance
    #config_result["raw_results"] = results[:, 1]
    config_result["mean_performance_final"] = mean_performance_final
    config_result["best_performance_final"] = best_performance_final
    config_result["std_perormance_final"] = std_perormance_final

    config_result["average_runtime"] = average_runtime
    print(config)
    print(f"result of {mean_performance_final}")
    return dataframe.append(config_result, ignore_index=True)

Optimal_config = {"pertubation_method": "Diagonal",
                    "annealing_alpha": 0.99,
                    "update_step_size_when_not_accepted_interval": 20,
                     "x_length": 5,
                     "x_bounds": (-500, 500),
                     "annealing_schedule":"simple_exponential_cooling",
                     "objective_function": rana_func,
                     "maximum_archive_length": 100,
                     "archive_minimum_acceptable_dissimilarity": 0.2,
                     "maximum_markov_chain_length": 9,
                     "maximum_function_evaluations": 10000,
                     "step_size_initialisation_fraction_of_range": 0.1,
                     "bound_enforcing_method": "not_clipping",
                     "cholesky_path_length": 5,
                    }

if __name__ == "__main__":
    method = "Rana"  # "both" # "Rosenbrock" # "Rana"
    if method == "Rana" or "both":
        df = pd.DataFrame()
        max_dim = 20
        all_configs = []
        for dim in range(1, max_dim + 1):
            Optimal_config["x_length"] = dim
            all_configs.append(Optimal_config.copy())
        with ProcessPoolExecutor() as executor:
            results = executor.map(run, all_configs)
        results_list = list(results)
        with open(f"./Simulated_Annealing_Param_Opt/stored_data/many_dim{time.time()}.pkl", "wb") as f:
            pickle.dump(results_list, f)
    elif method == "Rosenbrock" or "both":
        df = pd.DataFrame()
        max_dim = 10
        all_configs = []
        Optimal_config["objective_function"] = Rosenbrock
        for dim in range(1, max_dim + 1):
            Optimal_config["x_length"] = dim
            all_configs.append(Optimal_config.copy())
        with ProcessPoolExecutor() as executor:
            results = executor.map(run, all_configs)
        results_list = list(results)
        with open(f"./Simulated_Annealing_Param_Opt/stored_data/Rosenbrock__many_dim{time.time()}.pkl", "wb") as f:
            pickle.dump(results_list, f)

