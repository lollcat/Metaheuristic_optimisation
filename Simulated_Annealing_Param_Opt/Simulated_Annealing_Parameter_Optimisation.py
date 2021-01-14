import sys, os
sys.path.append(os.getcwd())    # need this for importing to work nicely

import pickle
import time
from Utils import running_mean, multiple_runs_with_different_seed
import numpy as np
import pandas as pd
from SimulatedAnealing import SimulatedAnnealing
from rana import rana_func
from Simulated_Annealing_Param_Opt.CONFIGS import Chol_config, Simple_config, Diag_config
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
    print("done")
    return dataframe.append(config_result, ignore_index=True)


if __name__ == "__main__":
    #selection = "diagonal"
    n_points = 50
    #run_func = lambda config: run(config, Class=SimulatedAnnealing, n_runs=n_runs)
    #run_func = lambda x: print(x)
    chol_configs = []
    diag_configs = []
    sim_configs = []
    annealing_alpha_z = list(np.linspace(0.5, 0.99, 10)) # list(1 - np.logspace(np.log(0.0001), 0.5, base=np.exp(1)))
    maximum_markov_chain_length_z = list(np.logspace(np.log10(5), np.log10(2000), n_points, dtype="int"))
    for maximum_markov_chain_length in maximum_markov_chain_length_z:
        for annealing_alpha in annealing_alpha_z:
            Chol_config["maximum_markov_chain_length"] = maximum_markov_chain_length
            Chol_config["annealing_alpha"] = annealing_alpha
            Diag_config["maximum_markov_chain_length"] = maximum_markov_chain_length
            Diag_config["annealing_alpha"] = annealing_alpha
            Simple_config["maximum_markov_chain_length"] = maximum_markov_chain_length
            Simple_config["annealing_alpha"] = annealing_alpha

            chol_configs.append(Chol_config.copy())
            diag_configs.append(Diag_config.copy())
            sim_configs.append(Simple_config.copy())
    with ProcessPoolExecutor() as executor:
        results = executor.map(run, chol_configs + diag_configs + sim_configs)
    results_list = list(results)
    with open(f"./Simulated_Annealing_Param_Opt/stored_data/alpha_markov_chain{time.time()}.pkl", "wb") as f:
        pickle.dump(results_list, f)
    """    
    if selection == "cholesky":
        with ProcessPoolExecutor() as executor:
            #results = executor.map(run_func, chol_configs)
            results = executor.map(run, chol_configs)
    elif selection == "diagonal":
        with ProcessPoolExecutor() as executor:
            results = executor.map(run, diag_configs)
            #results = executor.map(run, diag_configs)
    elif selection == "simple":
        with ProcessPoolExecutor() as executor:
            #results = executor.map(run_func, sim_configs)
            results = executor.map(run, sim_configs)
    results_list = list(results)
    with open(f"./Simlated_Annealing_Notebooks/stored_data/{selection}alpha_markov_chain{time.time()}.pkl", "wb") as f:
        pickle.dump(results_list, f)
    print("done")
    """
