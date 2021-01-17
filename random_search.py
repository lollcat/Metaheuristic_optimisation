import numpy as np
from rana import rana_func
import time

class random_search:
    def __init__(self, x_length, x_bounds= (-500, 500), objective_function=rana_func, n_evaluations=10000):
        self.x_length = x_length
        self.x_bounds = x_bounds
        self.objective_function = objective_function
        self.n_evaluations = n_evaluations

    def run(self):
        self.all_points = np.random.uniform(low=self.x_bounds[0], high=self.x_bounds[1], size=(self.n_evaluations, self.x_length))
        self.objectives = np.apply_along_axis(func1d=self.objective_function, arr=self.all_points, axis=1)
        return self.objectives.min(), self.all_points[np.argmin(self.objectives), :]

def run_with_multiple_seeds(dim, func, n_runs=20):
    objective_func_list = []
    runtimes = []
    for i in range(n_runs):
        start = time.time()
        grid_searcher = random_search(dim, objective_function=func)
        min_objective, min_X = grid_searcher.run()
        objective_func_list.append(min_objective)
        runtime = time.time() - start
        runtimes.append(runtime)
    return np.mean(objective_func_list), np.std(objective_func_list), np.mean(runtimes)




if __name__ == "__main__":
    np.random.seed(0)

    grid_searcher = random_search(5, [-500, 500], rana_func)
    min_objective, min_X = grid_searcher.run()
    print(f"minimum objective was {min_objective} in {grid_searcher.all_points.shape[0]} search points")