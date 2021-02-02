import sys, os
sys.path.append(os.getcwd())    # need this for importing to work nicely

import pickle
import time
from Utils import multiple_runs_with_different_seed
import numpy as np
import pandas as pd
from EvolutionStrategy import EvolutionStrategy
from Evolution_stratergy_parameter_search.CONFIGS import Comp_config, Diag_config, Simple_config
from concurrent.futures import ProcessPoolExecutor


def run(config,  Class=EvolutionStrategy, n_runs=30):
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


if __name__ == "__main__":
    df = pd.DataFrame()
    #n_runs = 2
    n_points = 30
    #run_func = lambda config: run(config, Class=EvolutionStrategy, n_runs=n_runs)
    all_configs = []
    number_of_offspring_z = np.linspace(20, 4000, n_points, dtype="int")
    child_to_parent_ratio_z = list(set(np.linspace(2, 200, n_points, dtype="int")))
    selection_methods = ["standard_mew_comma_lambda", "elitist"]
    for selection_method in selection_methods:
        for number_of_offspring in number_of_offspring_z:
            for child_to_parent_ratio in child_to_parent_ratio_z:
                parent_number = max(int(number_of_offspring/child_to_parent_ratio), 1)
                Comp_config["parent_number"] = parent_number
                Comp_config["selection_method"] = selection_method
                Comp_config["child_to_parent_ratio"] = child_to_parent_ratio
                Diag_config["parent_number"] = parent_number
                Diag_config["selection_method"] = selection_method
                Diag_config["child_to_parent_ratio"] = child_to_parent_ratio
                Simple_config["parent_number"] = parent_number
                Simple_config["selection_method"] = selection_method
                Simple_config["child_to_parent_ratio"] = child_to_parent_ratio
                all_configs.append(Comp_config.copy())
                all_configs.append(Diag_config.copy())
                all_configs.append(Simple_config.copy())
    with ProcessPoolExecutor() as executor:
        #results = executor.map(run_func, all_configs)
        results = executor.map(run, all_configs)
    results_list = list(results)
    with open(f"./Evolution_stratergy_parameter_search/stored_data/param_find{time.time()}.pkl", "wb") as f:
        pickle.dump(results_list, f)
