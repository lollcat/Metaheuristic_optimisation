from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time

def running_mean(x, N=100):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def multiple_runs_with_different_seed(Class, class_argument, n_iterations):
    results = []
    for i in range(n_iterations):
        np.random.seed(i)
        start_time = time.time()
        instantiated_class = Class(**class_argument)
        x_result, objective_result = instantiated_class.run()

        results.append([objective_result, instantiated_class.objective_history_array.min(), time.time() - start_time])
    return np.array(results) # final result, minimal result, runtime

"""
def multiple_runs_with_different_seed(Class, class_argument, n_iterations=10):
    def run_class(seed):
        np.random.seed(seed)
        return Class(**class_argument).run()

    run_class(0)

    with ProcessPoolExecutor(max_workers=8) as executor:
        result = executor.map(run_class, list(np.arange(n_iterations)))
    return result
"""


if __name__ == "__main__":
    from EvolutionStrategy import EvolutionStrategy
    np.random.seed(0)
    x_length = 5
    mutation_method = "diagonal" #"complex"    # "simple"
    selection_method = "elitist"  # "standard_mew_comma_lambda"
    from rana import rana_func

    x_max = 500
    x_min = -x_max
    class_argument = {"x_length" : x_length,
            "x_bounds" : (x_min, x_max),
            "objective_function" : rana_func,
            "mutation_method" : mutation_method,
            "selection_method" : selection_method}
    result = multiple_runs_with_different_seed(EvolutionStrategy, class_argument, n_iterations=2)