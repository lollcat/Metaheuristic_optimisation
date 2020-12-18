import numpy as np


class GeneticAlgorithm:
    def __init__(self, x_length, x_bounds, objective_function,
                 parent_number=10,
                 selection_method="standard_mew_comma_lambda", mutation_method = "simple",
                 recombination_method="global",
                 termination_min_abs_difference=1e-3,
                 **kwargs):
        self.x_length = x_length
        self.x_bounds = x_bounds
        self.objective_function_raw = objective_function
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.recombination_method = recombination_method
        self.termination_min_abs_difference = termination_min_abs_difference

        if mutation_method == "complex":
            # Mutation parameters - taken from slides, recommended by (Schwefel 1987)
            self.mutation_tau = 1/np.sqrt(2*np.sqrt(x_length))
            self.mutation_tau_dash = 1/np.sqrt(2*x_length)
            self.mutation_Beta = 0.0873
        elif mutation_method == "simple":
            self.standard_deviation_simple = kwargs["standard_deviation_fraction_of_range"]*(x_bounds[1] - x_bounds[0])

        self.parent_number = parent_number
        self.offspring_number = parent_number * 7  # parent to offspring ratio from slides (Schwefel 1987)

        self.parents = np.zeros((self.parent_number, self.x_length))    # zeros not used, this just shows the shape of the array representing the parents
        self.parent_objectives = np.zeros(self.parent_number)   # initialise to zeros to show the shape
        self.offspring = np.zeros((self.offspring_number, self.x_length))    # initialise to zeros to show shape
        self.offspring_objectives = np.zeros(self.offspring_number)

        self.objective_function_evaluation_count = 0  # initialise

    def objective_function(self, *args, **kwargs):
        self.objective_function_evaluation_count += 1   # increment by one everytime objective function is called
        return self.objective_function_raw(*args, **kwargs)

    def run(self):
        self.initialise_random_population()
        while True:  # loop until termination criteria is reached
            self.select_parents()

            # termination criteria
            if max(self.parent_objectives) - min(self.parent_objectives) < self.termination_min_abs_difference:
                break

            self.create_new_offspring()

        return self.parents[np.argmin(self.parent_objectives), :], min(self.parent_objectives)

    def initialise_random_population(self):
        self.offspring = np.random.uniform(low=self.x_bounds[0], high=self.x_bounds[1], size=(self.offspring_number, self.x_length))
        self.offspring_objectives = np.apply_along_axis(func1d=self.objective_function, arr=self.offspring, axis=1)
        if self.selection_method == "elitist":  # require pool including parents for select_parents function in this case
            self.parents = np.random.uniform(low=self.x_bounds[0], high=self.x_bounds[1],
                                               size=(self.parent_number, self.x_length))
            self.parent_objectives = np.apply_along_axis(func1d=self.objective_function, arr=self.parents, axis=1)

    def select_parents(self):
        if self.selection_method == "standard_mew_comma_lambda":
            # choose top values in linear time (np.argpartition doesn't sort top values amongst themselves)
            parent_indxs = np.argpartition(self.offspring_objectives, -self.parent_number)[-self.parent_number:]
            self.parents = self.offspring[parent_indxs, :]
            self.parent_objectives = self.offspring_objectives[parent_indxs]


    def create_new_offspring(self):
        # recombination
        if self.recombination_method == "global":
            # for each element in each child, inherit from a random parent
            child_recombination_indxs = np.random.choice(self.parent_number, replace=True,
                                           size= (self.offspring_number, self.x_length))
            offspring_pre_mutation = self.parents[child_recombination_indxs, np.arange(self.x_length)]

        # mutation
        if self.mutation_method == "simple":
            u_random_sample = np.random.normal(loc=0, scale=self.standard_deviation_simple,
                                               size=offspring_pre_mutation.shape)
            x_new = offspring_pre_mutation + u_random_sample
            self.offspring = np.clip(x_new, self.x_bounds[0], self.x_bounds[1])
            self.offspring_objectives = np.apply_along_axis(func1d=self.objective_function, arr=self.offspring, axis=1)

    def update_archive(self, x_new, objective_new):
        function_archive = [f_archive for x_archive, f_archive in self.archive]
        dissimilarity = [np.sqrt((x_archive - x_new).T @ (x_archive - x_new)) for x_archive, f_archive in self.archive]
        if min(dissimilarity) > self.archive_minimum_acceptable_dissimilarity:
            if len(self.archive) < self.archive_maximum_length:  # archive not full
                self.archive.append((x_new, objective_new))
            else:  # if archive is full
                if objective_new < min(function_archive):
                    self.archive[int(np.argmax(function_archive))] = (x_new, objective_new)  # replace worst solution
        else:    # new solution is close to another
            if objective_new < min(function_archive):
                self.archive[int(np.argmin(dissimilarity))] = (x_new, objective_new)  # replace most similar value
            else:
                similar_and_better = np.array([dissimilarity[i] < self.archive_similar_dissimilarity and \
                                      function_archive[i] > objective_new
                                      for i in range(len(self.archive))])
                if True in similar_and_better:
                    self.archive[np.where(similar_and_better == True)[0][0]] = (x_new, objective_new)



if __name__ == "__main__":
    np.random.seed(0)
    test = 1

    if test == "rana":  # rana function
        from rana import rana_func
        x_max = 500
        x_min = -x_max
        rana_2d = GeneticAlgorithm(x_length=2, x_bounds=(x_min, x_max), objective_function=rana_func,
                                   standard_deviation_fraction_of_range=0.05)
        x_result, objective_result = rana_2d.run()

    if test == 1:
        simple_objective = lambda x: x[0]**2 + np.sin(x[1])
        x_max = 10
        x_min = -x_max
        simple_evolve = GeneticAlgorithm(x_length=2, x_bounds=(x_min, x_max), objective_function=simple_objective,
                                   standard_deviation_fraction_of_range=0.05)
        x_result, objective_result = simple_evolve.run()
        print(f"x_result = {x_result} \n objective_result = {objective_result}")

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        n = 100
        x1_linspace = np.linspace(-10, 10, n)
        x2_linspace = np.linspace(-10, 10, n)
        z = np.zeros((n, n))
        for i, x1_val in enumerate(x1_linspace):
            for j, x2_val in enumerate(x2_linspace):
                z[i, j] = simple_objective(np.array([x1_val, x2_val]))
        x1, x2 = np.meshgrid(x1_linspace, x2_linspace)
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x1.flatten(), x2.flatten(), z.flatten(), cmap=mpl.cm.jet)
        ax.plot(x_result[0], x_result[1], objective_result, "or")
        ax.set_xlabel("variable 1")
        ax.set_ylabel("variable 2")
        ax.set_zlabel("cost")
        fig.show()