import numpy as np
import itertools

class grid_search:
    def __init__(self, x_length, x_bounds, objective_function, n_evaluations=10000.0):
        self.x_length = x_length
        self.x_bounds = x_bounds
        self.objective_function = objective_function
        self.n_evaluations = n_evaluations

    def run(self):
        x_linspace = np.linspace(self.x_bounds[0], self.x_bounds[1], int(self.n_evaluations**(1/self.x_length)))
        self.all_points = np.array(list(itertools.product(x_linspace, repeat=self.x_length)))  # get cartesian product for x_linspace)
        self.objectives = np.apply_along_axis(func1d=self.objective_function, arr=self.all_points, axis=1)
        return self.objectives.min(), self.all_points[np.argmin(self.objectives), :]

if __name__ == "__main__":
    np.random.seed(0)

    from rana import rana_func
    grid_searcher = grid_search(5, [-500, 500], rana_func)
    min_objective, min_X = grid_searcher.run()
    print(f"minimum objective was {min_objective} in {grid_searcher.all_points.shape[0]} search points")

