import numpy as np


class EvolutionStrategy:
    def __init__(self, x_length, x_bounds, objective_function, archive_minimum_acceptable_dissimilarity,
                 parent_number=10,
                 selection_method="standard_mew_comma_lambda", mutation_method = "simple",
                 recombination_method="global",
                 termination_min_abs_difference=1e-3,
                 maximum_archive_length=30, objective_count_maximum=10000,
                 mutation_covariance_initialisation_fraction_of_range=0.1,
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
            self.mutation_standard_deviations = np.ones(x_length) * \
                                                mutation_covariance_initialisation_fraction_of_range * (x_bounds[1] - x_bounds[0])

            self.rotation_matrix = np.zeros(x_length, x_length)
        elif mutation_method == "simple":
            self.standard_deviation_simple = mutation_covariance_initialisation_fraction_of_range*(x_bounds[1] - x_bounds[0])

        self.parent_number = parent_number
        self.offspring_number = parent_number * 7  # parent to offspring ratio from slides (Schwefel 1987)

        self.parents = np.zeros((self.parent_number, self.x_length))    # zeros not used, this just shows the shape of the array representing the parents
        self.parent_objectives = np.zeros(self.parent_number)   # initialise to zeros to show the shape
        self.offspring = np.zeros((self.offspring_number, self.x_length))    # initialise to zeros to show shape
        self.offspring_objectives = np.zeros(self.offspring_number)

        self.objective_function_evaluation_count = 0  # initialise
        self.objective_count_maximum = objective_count_maximum


        # initialise archive and parameters determining how archive is managed
        self.archive = []   # list of (x, objective value) tuples
        self.archive_maximum_length = maximum_archive_length
        self.archive_minimum_acceptable_dissimilarity = archive_minimum_acceptable_dissimilarity
        self.archive_similar_dissimilarity = archive_minimum_acceptable_dissimilarity
        self.objective_history = []

    def objective_function(self, *args, **kwargs):
        self.objective_function_evaluation_count += 1   # increment by one everytime objective function is called
        return self.objective_function_raw(*args, **kwargs)

    def run(self):
        self.initialise_random_population()
        while True:  # loop until termination criteria is reached
            self.select_parents()
            for x, objective in zip(self.parents, self.parent_objectives):  # update archive
                self.update_archive(x, objective)
                self.objective_history.append(objective)
            # termination criteria
            if max(self.parent_objectives) - min(self.parent_objectives) < self.termination_min_abs_difference \
                    or self.objective_function_evaluation_count > self.objective_count_maximum:
                break

            self.create_new_offspring()

        return self.parents[np.argmin(self.parent_objectives), :], min(self.parent_objectives)

    def initialise_random_population(self):
        self.offspring = np.random.uniform(low=self.x_bounds[0], high=self.x_bounds[1], size=(self.offspring_number, self.x_length))
        self.offspring_objectives = np.squeeze(np.apply_along_axis(func1d=self.objective_function, arr=self.offspring, axis=1))
        if self.selection_method == "elitist":  # require pool including parents for select_parents function in this case
            self.parents = np.random.uniform(low=self.x_bounds[0], high=self.x_bounds[1],
                                               size=(self.parent_number, self.x_length))
            self.parent_objectives = np.apply_along_axis(func1d=self.objective_function, arr=self.parents, axis=1)

    def select_parents(self):
        if self.selection_method == "standard_mew_comma_lambda":
            # choose top values in linear time (np.argpartition doesn't sort top values amongst themselves)
            new_parent_indxs = np.argpartition(self.offspring_objectives, self.parent_number)[:self.parent_number]
            self.parents = self.offspring[new_parent_indxs, :]
            self.parent_objectives = self.offspring_objectives[new_parent_indxs]
        elif self.selection_method == "standard_mew_plus_lambda":
            # create pool selcted from
            pool_objectives = np.zeros(self.parent_number + self.offspring_number)
            pool_objectives[0:self.offspring_number] = self.offspring_objectives
            pool_objectives[self.offspring_number:] = self.parent_objectives
            pool = np.zeros(self.offspring_number + self.parent_objectives, self.x_length)
            pool[0:self.offspring_number, :] = self.offspring
            pool[self.offspring_number:, :] = self.parents

            new_parent_indxs = np.argpartition(pool_objectives, self.parent_number)[:self.parent_number]
            self.parents = self.offspring[new_parent_indxs, :]
            self.parent_objectives = pool_objectives[new_parent_indxs]

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
            self.offspring_objectives = np.squeeze(np.apply_along_axis(func1d=self.objective_function, arr=self.offspring, axis=1))
        elif self.mutation_method == "complex": # non isotropic covariance
            self.mutation_standard_deviations = self.mutation_standard_deviations * \
                                                np.exp(self.mutation_tau_dash*np.random.normal(0,1)
                                                       +self.mutation_tau*np.random.normal(0,1, size=self.x_length))
            self.rotation_matrix = self.rotation_matrix + self.mutation_Beta*np.random.normal(0,1, size=(self.x_length, self.x_length))
            #rotation_matrix = 1/2*np.arctan(2 * np.divide(self.mutation_covariance,
            #                                              np.einsum("ij,jk->ik",
            #                                                        self.mutation_standard_deviations[:, np.newaxis] ** 2,
            #                                                        -self.mutation_standard_deviations[np.newaxis, :] ** 2)))
            if not np.all(np.linalg.eigvals(rotation_matrix) > 0):
                rotation_matrix += np.eye(self.x_length)*1e-16
            np.random.multivariate_normal(mean=np.zeros(self.x_length), cov=np.linalg.inv(rotation_matrix)@rotation_matrix,
                                          size=self.offspring_number)


    def update_archive(self, x_new, objective_new):
        if len(self.archive) == 0:  # if empty then initialise with the first value
            self.archive.append((x_new, objective_new))
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
    test = "rana"
    import matplotlib.pyplot as plt

    if test == "rana":  # rana function
        from rana import rana_func
        x_max = 500
        x_min = -x_max
        rana_2d = EvolutionStrategy(x_length=2, x_bounds=(x_min, x_max), objective_function=rana_func,
                                    standard_deviation_fraction_of_range=0.001,
                                    archive_minimum_acceptable_dissimilarity=20)
        x_result, objective_result = rana_2d.run()
        plt.plot(rana_2d.objective_history)
        plt.show()

    if test == 0:   # simplest objective
        x_max = 50
        x_min = -x_max
        simple_objective = lambda x: x + np.sin(x)*20 + 3
        simple_evolve = EvolutionStrategy(x_length=1, x_bounds=(x_min, x_max), objective_function=simple_objective,
                                          standard_deviation_fraction_of_range=0.05,
                                          archive_minimum_acceptable_dissimilarity=5)
        x_result, objective_result = simple_evolve.run()
        print(f"x_result = {x_result} \n objective_result = {objective_result}")

        archive_x = np.array([x_archive for x_archive, f_archive in simple_evolve.archive])
        archive_f = np.array([f_archive for x_archive, f_archive in simple_evolve.archive])


        x_linspace = np.linspace(x_min, x_max, 200)
        plt.plot(x_linspace, simple_objective(x_linspace))
        plt.plot(x_result, objective_result, "or")
        plt.plot(archive_x, archive_f, "xr")
        plt.show()

        plt.plot(simple_evolve.objective_history)
        plt.show()