import sys, os
sys.path.append(os.getcwd())    # need this for importing to work nicely

import pickle
import time
from Utils import multiple_runs_with_different_seed
import numpy as np
import pandas as pd
from EvolutionStrategy import EvolutionStrategy
from concurrent.futures import ProcessPoolExecutor
from rana import rana_func


def run(config,  Class=EvolutionStrategy, n_runs=2):
        dataframe = pd.DataFrame()
        results = multiple_runs_with_different_seed(Class=Class, class_argument=config, n_iterations=n_runs,
                                                    method="ES")
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
        return dataframe.append(config_result, ignore_index=True)

Optimal_config = {"objective_function": rana_func,
               "x_bounds" : (-500, 500),
                "x_length" : 5,
                "parent_number" : 23,
                "child_to_parent_ratio" : 11,
                "bound_enforcing_method" : "not_clipping" ,
                "selection_method" :  "standard_mew_comma_lambda" ,
                "standard_deviation_clipping_fraction_of_range" : 0.01,
                "mutation_covariance_initialisation_fraction_of_range" : 0.01 ,
                "mutation_method" : "diagonal",
               "termination_min_abs_difference": 1e-6,
                  "maximum_archive_length": 100, "archive_minimum_acceptable_dissimilarity":0.1}

if __name__ == "__main__":
    df = pd.DataFrame()
    max_dim = 10
    all_configs = []
    for dim in range(1, max_dim + 1):
        Optimal_config["x_length"] = dim
        all_configs.append(Optimal_config.copy())
    with ProcessPoolExecutor() as executor:
        results = executor.map(run, all_configs)
    results_list = list(results)
    with open(f"./Evolution_stratergy_parameter_search/stored_data/many_dim{time.time()}.pkl", "wb") as f:
        pickle.dump(results_list, f)
