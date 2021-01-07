import numpy as np


class EvolutionStrategy:
    def __init__(self, x_length, x_bounds, objective_function, archive_minimum_acceptable_dissimilarity=0.1,
                 parent_number=10,
                 selection_method="standard_mew_comma_lambda", mutation_method = "simple",
                 recombination_method="global",
                 termination_min_abs_difference=1e-6,
                 maximum_archive_length=30, objective_count_maximum=10000,
                 mutation_covariance_initialisation_fraction_of_range=0.01,
                 standard_deviation_clipping_fraction_of_range = 0.05,
                 bound_enforcing_method="not_clipping",
                 child_to_parent_ratio=7,
                 **kwargs):
        self.x_length = x_length
        self.x_bounds = x_bounds
        self.bound_enforcing_method = bound_enforcing_method
        self.x_range = 2  # from -1 to 1 when interpolated
        self.objective_function_raw = objective_function
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.recombination_method = recombination_method
        self.termination_min_abs_difference = termination_min_abs_difference
        self.standard_deviation_clipping_fraction_of_range = standard_deviation_clipping_fraction_of_range

        self.parent_number = parent_number
        self.offspring_number = parent_number * child_to_parent_ratio  # parent to offspring ratio from slides (Schwefel 1987)

        if mutation_method == "complex":
            # Mutation parameters - taken from slides, recommended by (Schwefel 1987)
            self.mutation_tau = 1/np.sqrt(2*np.sqrt(self.x_length))
            self.mutation_tau_dash = 1/np.sqrt(2*self.x_length)
            self.mutation_Beta = 0.0873
            self.offspring_mutation_standard_deviations = np.ones((self.offspring_number, self.x_length)) * \
                                                          mutation_covariance_initialisation_fraction_of_range * self.x_range
            self.parent_mutation_standard_deviations = np.ones((self.parent_number, self.x_length)) * \
                                                       mutation_covariance_initialisation_fraction_of_range * self.x_range

            self.offspring_rotation_matrices = np.broadcast_to(np.eye(self.x_length), (self.offspring_number, self.x_length, self.x_length))

            self.make_covariance_matrix()
            self.parent_rotation_matrices = self.offspring_rotation_matrices[0:self.parent_number, :, :]    # just slice children for initialisation
            self.parent_covariance_matrices = self.offspring_covariance_matrices[0:self.parent_number, :, :]
        elif mutation_method == "simple":
            self.standard_deviation_simple = mutation_covariance_initialisation_fraction_of_range*self.x_range

        elif mutation_method == "diagonal":
            self.mutation_tau = 1 / np.sqrt(2 * np.sqrt(self.x_length))
            self.mutation_tau_dash = 1 / np.sqrt(2 * self.x_length)
            self.offspring_mutation_standard_deviations = np.ones((self.offspring_number, self.x_length)) * \
                                                          mutation_covariance_initialisation_fraction_of_range * self.x_range
            self.parent_mutation_standard_deviations = np.ones((self.parent_number, self.x_length)) * \
                                                mutation_covariance_initialisation_fraction_of_range * self.x_range



        self.parents = np.zeros((self.parent_number, self.x_length))    # zeros not used, this just shows the shape of the array representing the parents
        self.parent_objectives = np.zeros(self.parent_number)   # initialise to zeros to show the shape
        self.offspring = np.zeros((self.offspring_number, self.x_length))    # initialise to zeros to show shape
        self.offspring_objectives = np.zeros(self.offspring_number)

        self.objective_function_evaluation_count = 0  # initialise
        self.generation_number = 0
        self.objective_count_maximum = objective_count_maximum


        # initialise archive and parameters determining how archive is managed
        self.archive = []   # list of (x, objective value) tuples
        self.archive_maximum_length = maximum_archive_length
        self.archive_minimum_acceptable_dissimilarity = archive_minimum_acceptable_dissimilarity
        self.archive_similar_dissimilarity = archive_minimum_acceptable_dissimilarity
        self.parent_objective_history = []
        self.parent_standard_deviation_history = []
        self.offspring_objective_history = []

    def objective_function(self, x):
        # interpolation done here to pass the objective function x correctly interpolated
        self.objective_function_evaluation_count += 1   # increment by one everytime objective function is called
        x_interp = np.interp(x, [-1, 1], self.x_bounds)
        result = self.objective_function_raw(x_interp)
        return result

    def run(self):
        self.initialise_random_population()
        while True:  # loop until termination criteria is reached
            self.generation_number += 1
            self.select_parents()
            #self.objective_history.append([self.parent_objectives.min(), self.parent_objectives.mean()])
            for x, objective in zip(self.parents, self.parent_objectives):  # update archive
                self.update_archive(x, objective)
            self.parent_objective_history.append(self.parent_objectives)
            if self.mutation_method == "diagonal" or self.mutation_method == "complex":
                self.parent_standard_deviation_history.append(self.parent_mutation_standard_deviations)
            if self.objective_function_evaluation_count > self.objective_count_maximum:
                print("max total iterations")
                break
            """
            # termination criteria
            if max(self.parent_objectives) - min(self.parent_objectives) < self.termination_min_abs_difference:
                print("converged")
                break
            """
            self.create_new_offspring()
            self.offspring_objective_history.append(self.offspring_objectives)

        return self.parents[np.argmin(self.parent_objectives), :], min(self.parent_objectives)

    def initialise_random_population(self):
        self.offspring = np.random.uniform(low=-1, high=1, size=(self.offspring_number, self.x_length))
        self.offspring_objectives = np.squeeze(np.apply_along_axis(func1d=self.objective_function, arr=self.offspring, axis=1))
        if self.selection_method == "elitist":  # require pool including parents for select_parents function in this case
            self.parents = np.random.uniform(low=-1, high=1,
                                               size=(self.parent_number, self.x_length))
            self.parent_objectives = np.apply_along_axis(func1d=self.objective_function, arr=self.parents, axis=1)

    def select_parents(self):
        if self.selection_method == "standard_mew_comma_lambda":
            # choose top values in linear time (np.argpartition doesn't sort top values amongst themselves)
            pool_objectives = self.offspring_objectives
            pool = self.offspring
            if self.mutation_method == "diagonal":
                pool_standard_deviations = self.offspring_mutation_standard_deviations
            if self.mutation_method == "complex":
                pool_standard_deviations = self.offspring_mutation_standard_deviations
                pool_rotation_matrices = self.offspring_rotation_matrices
        else:
            assert self.selection_method == "elitist"
            # create pool selcted from
            pool_objectives = np.zeros(self.parent_number + self.offspring_number)
            pool_objectives[0:self.offspring_number] = self.offspring_objectives
            pool_objectives[self.offspring_number:] = self.parent_objectives
            pool = np.zeros((self.offspring_number + self.parent_number, self.x_length))
            pool[0:self.offspring_number, :] = self.offspring
            pool[self.offspring_number:, :] = self.parents
            if self.mutation_method == "diagonal":
                pool_standard_deviations = np.zeros((self.offspring_number + self.parent_number, self.x_length))
                pool_standard_deviations[0:self.offspring_number, :] = self.offspring_mutation_standard_deviations
                pool_standard_deviations[self.offspring_number:, :] = self.parent_mutation_standard_deviations
            if self.mutation_method == "complex":
                pool_standard_deviations = np.zeros((self.offspring_number + self.parent_number, self.x_length))
                pool_standard_deviations[0:self.offspring_number, :] = self.offspring_mutation_standard_deviations
                pool_standard_deviations[self.offspring_number:, :] = self.parent_mutation_standard_deviations
                pool_rotation_matrices = np.zeros((self.offspring_number + self.parent_number, self.x_length, self.x_length))
                pool_rotation_matrices[0:self.offspring_number, :, :] = self.offspring_rotation_matrices
                pool_rotation_matrices[self.offspring_number:, :, :] = self.parent_rotation_matrices


        new_parent_indxs = np.argpartition(pool_objectives, self.parent_number)[:self.parent_number]
        self.parents = pool[new_parent_indxs, :]
        self.parent_objectives = pool_objectives[new_parent_indxs]

        if self.mutation_method == "diagonal":
            self.parent_mutation_standard_deviations = pool_standard_deviations[new_parent_indxs, :]
        elif self.mutation_method == "complex":
            self.parent_mutation_standard_deviations = pool_standard_deviations[new_parent_indxs, :]
            self.parent_rotation_matrices = pool_rotation_matrices[new_parent_indxs, :, :]


    def create_new_offspring(self):
        # recombination
        if self.recombination_method == "global":
            # for each element in each child, inherit from a random parent
            child_recombination_indxs = np.random.choice(self.parent_number, replace=True,
                                           size = (self.offspring_number, self.x_length))
            offspring_pre_mutation = self.parents[child_recombination_indxs, np.arange(self.x_length)]
            if self.mutation_method == "diagonal" or self.mutation_method == "complex":
                #  offspring_pre_mutation_standard_deviation = self.parent_mutation_standard_deviations[
                #     child_recombination_indxs, np.arange(self.x_length)]
                child_stratergy_recombination_indxs = np.random.choice(self.parent_number, replace=True,
                                           size=(self.offspring_number,  self.x_length, 2))
                offspring_pre_mutation_standard_deviation = \
                    0.5*self.parent_mutation_standard_deviations[
                        child_stratergy_recombination_indxs[:, :, 0], np.arange(self.x_length)] +\
                    0.5*self.parent_mutation_standard_deviations[
                        child_stratergy_recombination_indxs[:, :, 1], np.arange(self.x_length)]
                if self.mutation_method == "complex":
                    child_stratergy_recombination_indxs = np.random.choice(self.parent_number, replace=True,
                                                                           size=(
                                                                           self.offspring_number*self.x_length*self.x_length, 2))
                    slices1 = np.broadcast_to(np.arange(self.x_length)[np.newaxis, :], (self.x_length, self.x_length)).flatten()
                    slices1 = np.broadcast_to(slices1[np.newaxis, :], (self.offspring_number, self.x_length**2)).flatten()
                    slices2 = np.broadcast_to(np.arange(self.x_length)[:, np.newaxis], (self.x_length, self.x_length)).flatten()
                    slices2 = np.broadcast_to(slices2[np.newaxis, :], (self.offspring_number, self.x_length**2)).flatten()
                    offspring_pre_mutation_rotation_matrix = \
                        np.reshape(0.5 * self.parent_rotation_matrices[
                            child_stratergy_recombination_indxs[:, 0], slices1,
                            slices2 ] + \
                        0.5 * self.parent_rotation_matrices[
                            child_stratergy_recombination_indxs[:, 1], slices1,
                            slices2], (self.offspring_number, self.x_length, self.x_length))


        # mutation
        if self.mutation_method == "simple":
            u_random_sample = np.random.normal(loc=0, scale=self.standard_deviation_simple,
                                               size=offspring_pre_mutation.shape)
            x_new = offspring_pre_mutation + u_random_sample
            if self.bound_enforcing_method == "clipping":
                x_new = np.clip(x_new, -1, 1)
            else:
                while np.max(x_new) > 1 or np.min(x_new) < -1:
                    indxs_breaking_bounds = np.where((x_new > 1) + (x_new < -1) == 1)
                    u_random_sample = np.random.normal(loc=0, scale=self.standard_deviation_simple,
                                                       size=indxs_breaking_bounds[0].size)
                    x_new[indxs_breaking_bounds ] = offspring_pre_mutation[indxs_breaking_bounds ] + u_random_sample


        elif self.mutation_method == "diagonal":  # non spherical covariance
            self.offspring_mutation_standard_deviations = \
                offspring_pre_mutation_standard_deviation * \
                np.exp(
                        self.mutation_tau_dash*np.broadcast_to(
                            np.random.normal(0, 1, size=(self.offspring_number, 1)), self.offspring_mutation_standard_deviations.shape)
                           + self.mutation_tau*np.random.normal(
                                               0, 1, size=self.offspring_mutation_standard_deviations.shape))
            self.offspring_mutation_standard_deviations = np.clip(self.offspring_mutation_standard_deviations,
                                                                  1e-8, self.standard_deviation_clipping_fraction_of_range*self.x_range)

            u_random_sample = np.random.normal(loc=0, scale=self.offspring_mutation_standard_deviations,
                                               size=offspring_pre_mutation.shape)
            x_new = offspring_pre_mutation + u_random_sample
            if self.bound_enforcing_method == "clipping":
                x_new = np.clip(x_new, -1, 1)
            else:
                while np.max(x_new) > 1 or np.min(x_new) < -1:
                    indxs_breaking_bounds = np.where((x_new > 1) + (x_new < -1) == 1)
                    u_random_sample = np.random.normal(loc=0, scale=self.offspring_mutation_standard_deviations[indxs_breaking_bounds],
                                                       size=indxs_breaking_bounds[0].size)
                    x_new[indxs_breaking_bounds] = offspring_pre_mutation[indxs_breaking_bounds] + u_random_sample

        if self.mutation_method == "complex":
            self.offspring_mutation_standard_deviations = \
                offspring_pre_mutation_standard_deviation  * \
                np.exp(
                        self.mutation_tau_dash*np.broadcast_to(
                            np.random.normal(0, 1, size=(self.offspring_number, 1)), self.offspring_mutation_standard_deviations.shape)
                           + self.mutation_tau*np.random.normal(
                                               0, 1, size=self.offspring_mutation_standard_deviations.shape))

            self.offspring_mutation_standard_deviations = np.clip(self.offspring_mutation_standard_deviations,
                                                                  1e-8,
                                                                  self.standard_deviation_clipping_fraction_of_range * self.x_range)

            self.offspring_rotation_matrices = offspring_pre_mutation_rotation_matrix + self.mutation_Beta * np.random.normal(0, 1, size=(self.offspring_rotation_matrices.shape))
            #self.offspring_rotation_matrices = self.offspring_rotation_matrices/np.broadcast_to(np.linalg.det(self.offspring_rotation_matrices)[:, np.newaxis, np.newaxis], (self.offspring_number, self.x_length, self.x_length))
            #self.offspring_rotation_matrices = np.clip(self.offspring_rotation_matrices, -np.pi/4, np.pi/4)

            # rotation_matrix = 1/2*np.arctan(2 * np.divide(self.mutation_covariance,
            #                                               np.einsum("ij,jk->ik",
            #                                                         self.mutation_standard_deviations[:, np.newaxis] ** 2,
            #                                                         -self.mutation_standard_deviations[np.newaxis, :] ** 2)))
            for i in range(self.offspring_number):
                self.offspring_rotation_matrices[i, :, :] = np.tril(self.offspring_rotation_matrices[i, :, :], k=-1) - np.tril(
                    self.offspring_rotation_matrices[i, :, :], k=-1).T      # make symmetric
            self.make_covariance_matrix()
            for i in range(self.offspring_number):
                covariance_matrix = self.make_positive_definate(self.offspring_covariance_matrices[i, :, :])
                self.offspring[i, :] = offspring_pre_mutation[i, :] + np.random.multivariate_normal(mean=np.zeros(self.x_length), cov=covariance_matrix)
                if self.bound_enforcing_method == "clipping":
                    self.offspring[i, :]  =  np.clip(self.offspring[i, :] , -1, 1)
                else:
                    if np.max(self.offspring[i, :]) > 1 or np.min(self.offspring[i, :]) < -1:
                        self.offspring[i, :] = offspring_pre_mutation[i, :] + np.random.multivariate_normal(mean=np.zeros(self.x_length), cov=covariance_matrix)
        if self.mutation_method != "complex":
            self.offspring = x_new
        self.offspring_objectives = np.squeeze(
            np.apply_along_axis(func1d=self.objective_function, arr=self.offspring, axis=1))

    def make_positive_definate(self, matrix, i=1):
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except:
            if i > 10:
                raise Exception("matrix unable to be made positive definate")
            matrix += np.eye(self.x_length) * 1e-6 * 10**i
            return self.make_positive_definate(matrix, i=i+1)

        """
            matrix += np.eye(self.x_length) * 2e-1
            try:
                np.linalg.cholesky(matrix)
                return matrix
            except:
                print("matrix unable to be made positive definate")
        """


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

    def make_covariance_matrix(self):
        sigma_i = np.zeros((self.offspring_number, self.x_length, self.x_length))
        sigma_j = np.zeros((self.offspring_number, self.x_length, self.x_length))
        for offspring_number in range(self.offspring_number):
            stds = self.offspring_mutation_standard_deviations[offspring_number, :]
            sigma_i[offspring_number, np.arange(self.x_length), :] = stds
            sigma_j[offspring_number, :, np.arange(self.x_length)] = stds
        self.offspring_covariance_matrices = np.tan(2 * self.offspring_rotation_matrices) * (sigma_i**2 - sigma_j**2) * 1/2
        self.offspring_covariance_matrices = np.clip(self.offspring_covariance_matrices, -np.minimum(sigma_i, sigma_j), np.minimum(sigma_i, sigma_j))
        #self.offspring_covariance_matrices = np.clip(self.offspring_covariance_matrices,
        #                                            -self.x_range*self.standard_deviation_clipping_fraction_of_range,
         #                                           self.standard_deviation_clipping_fraction_of_range*self.x_range)
        self.offspring_covariance_matrices[:, np.arange(self.x_length), np.arange(self.x_length)] = self.offspring_mutation_standard_deviations

    @property
    def parent_objective_history_array(self):
        return np.array(self.parent_objective_history)

    @property
    def offspring_objective_history_array(self):
        return np.array(self.offspring_objective_history)

    @property
    def parent_standard_deviation_history_array(self):
        return np.array(self.parent_standard_deviation_history)



if __name__ == "__main__":
    np.random.seed(0)
    x_length = 5
    mutation_method = "complex"  # "diagonal" #"complex"    # "simple"
    selection_method =  "elitist" # "standard_mew_comma_lambda" #  "elitist"  # "standard_mew_comma_lambda"
    clipping_method = "not_clipping"
    from rana import rana_func

    x_max = 500
    x_min = -x_max
    rana_2d = EvolutionStrategy(x_length=x_length, x_bounds=(x_min, x_max), objective_function=rana_func,
                                mutation_method=mutation_method, selection_method =selection_method,
                                bound_enforcing_method=clipping_method, parent_number=10)
    x_result, objective_result = rana_2d.run()
    print(f"x_result = {x_result} \n objective_result = {objective_result} \n ")

    """
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
"""