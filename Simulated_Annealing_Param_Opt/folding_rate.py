import sys, os
sys.path.append(os.getcwd())    # need this for importing to work nicely

import pickle
import time
from Utils import running_mean, multiple_runs_with_different_seed
import numpy as np
import pandas as pd
from SimulatedAnealing import SimulatedAnnealing
from Simulated_Annealing_Param_Opt.Simulated_Annealing_Parameter_Optimisation import run
from rana import rana_func
from Simulated_Annealing_Param_Opt.CONFIGS import Chol_config, Simple_config, Diag_config
from concurrent.futures import ProcessPoolExecutor

if __name__ == "__main__":
    n_points = 30
    chol_configs = []
    diag_configs = []
    fold_rates = list(set(np.logspace(np.log10(1), np.log10(10000), n_points, dtype="int")))
    for update_step_size_when_not_accepted_interval in fold_rates:
        Chol_config["update_step_size_when_not_accepted_interval"] = update_step_size_when_not_accepted_interval
        Diag_config["update_step_size_when_not_accepted_interval"] = update_step_size_when_not_accepted_interval
        chol_configs.append(Chol_config.copy())
        diag_configs.append(Diag_config.copy())
    with ProcessPoolExecutor() as executor:
        results = executor.map(run, chol_configs + diag_configs)
    results_list = list(results)
    with open(f"./Simulated_Annealing_Param_Opt/stored_data/folding{time.time()}.pkl", "wb") as f:
        pickle.dump(results_list, f)