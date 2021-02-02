import numpy as np


class EvolutionStrategy:
    """
    x is transformed from the original bounds, to be bounded by (-1, 1)
    Before outputs are given x is re-transformed back into the original form
    """
    def __init__(self, x_length, x_bounds, objective_function, archive_minimum_acceptable_dissimilarity=0.1,
                 parent_number=10,
                 selection_method="standard_mew_comma_lambda", mutation_method = "simple",
                 recombination_method="global",
                 termination_min_abs_difference=1e-6,
                 maximum_archive_length=None, objective_count_maximum=10000,
                 mutation_covariance_initialisation_fraction_of_range=0.01,
                 standard_deviation_clipping_fraction_of_range = 0.05,
                 bound_enforcing_method="not_clipping",
                 child_to_parent_ratio=7):
        self.x_length = x_length
        self.x_bounds = x_bounds
        self.bound_enforcing_method = bound_enforcing_method
        self.x_range = 2  # after being transformed to be between -1 and 1
        self.objective_function_raw = objective_function
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.recombination_method = recombination_method
        self.termination_min_abs_difference = termination_min_abs_difference

        # prevent standard deviations from becoming too large (clip relative to size of range)
        self.standard_deviation_clipping_fraction_of_range = standard_deviation_clipping_fraction_of_range
        self.parent_number = parent_number
        self.offspring_number = parent_number * child_to_parent_ratio

        if mutation_method == "complex":
            # Mutation parameters - taken from slides, recommended by (Schwefel 1987)
            self.mutation_tau = 1/np.sqrt(2*np.sqrt(self.x_length))
            self.mutation_tau_dash = 1/np.sqrt(2*self.x_length)
            self.mutation_Beta = 0.0873
            self.offspring_mutation_standard_deviations = \
                np.ones((self.offspring_number, self.x_length)) * \
                mutation_covariance_initialisation_fraction_of_range * self.x_range
            self.parent_mutation_standard_deviations = \
                np.ones((self.parent_number, self.x_length)) * \
                mutation_covariance_initialisation_fraction_of_range * self.x_range

            self.offspring_rotation_matrices = np.broadcast_to(np.eye(self.x_length), (self.offspring_number, self.x_length, self.x_length))

            self.make_covariance_matrix()
            # just slice children for initialisation
            self.parent_rotation_matrices = self.offspring_rotation_matrices[0:self.parent_number, :, :]
            self.parent_covariance_matrices = self.offspring_covariance_matrices[0:self.parent_number, :, :]


        elif mutation_method == "simple":
            self.standard_deviation_simple = mutation_covariance_initialisation_fraction_of_range*self.x_range

        elif mutation_method == "diagonal":
            self.mutation_tau = 1 / np.sqrt(2 * np.sqrt(self.x_length))
            self.mutation_tau_dash = 1 / np.sqrt(2 * self.x_length)
            self.offspring_mutation_standard_deviations = \
                np.ones((self.offspring_number, self.x_length)) * \
                mutation_covariance_initialisation_fraction_of_range * self.x_range
            self.parent_mutation_standard_deviations = np.ones((self.parent_number, self.x_length)) * mutation_covariance_initialisation_fraction_of_range * self.x_range

        # initialise parents and offspring
        # zeros aren't ever used, just specifies the shapes of the arrays
        self.parents = np.zeros((self.parent_number, self.x_length))
        self.parent_objectives = np.zeros(self.parent_number)
        self.offspring = np.zeros((self.offspring_number, self.x_length))
        self.offspring_objectives = np.zeros(self.offspring_number)


        # initialise archive and parameters determining how archive is managed
        self.archive = []   # list of (x, objective value) tuples
        self.archive_maximum_length = maximum_archive_length    # If none then don't store, as slows program down slightly
        self.archive_minimum_acceptable_dissimilarity = archive_minimum_acceptable_dissimilarity
        self.archive_similar_dissimilarity = archive_minimum_acceptable_dissimilarity

        # initialise histories and counters
        # these are useful for inspecting the performance of the algorithm after a run
        # and are used within the program (e.g. markov chain length)
        self.parent_objective_history = []
        self.parent_x_history = []
        self.parent_standard_deviation_history = []
        self.offspring_objective_history = []
        self.offspring_x_history = []
        self.parent_covariance_determinant_history = []
        self.offspring_covariance_determinant_history = []
        self.objective_function_evaluation_count = 0  # initialise
        self.generation_number = 0
        self.objective_function_evaluation_max_count = objective_count_maximum

    def objective_function(self, x):
        """
        Wrapper for the objective function calls, adding some extra functionality
        """
        # increment by one everytime objective function is called
        self.objective_function_evaluation_count += 1
        # interpolation done here to pass the objective function x correctly interpolated
        x_interp = np.interp(x, [-1, 1], self.x_bounds)
        result = self.objective_function_raw(x_interp)
        return result

    def run(self):
        """
        This function run's the major steps of the Evolution Strategy algorithm
        # major steps in the algorithm are surrounded with a #*******************************
        # other parts of this function are more organisation (e.g. storing histories)
        """
        self.initialise_random_population()
        while True:  # loop until termination criteria is reached
            self.generation_number += 1
            # **************************   Selection *******************
            self.select_parents()
            # **************************************************************
            if self.archive_maximum_length is not None: # if archive is None, then don't store
                for x, objective in zip(self.parents, self.parent_objectives):  # update archive
                    self.update_archive(x, objective)
            self.parent_objective_history.append(self.parent_objectives)
            self.parent_x_history.append(self.parents)
            if self.mutation_method == "diagonal":
                self.parent_standard_deviation_history.append\
                    (self.parent_mutation_standard_deviations)
                self.parent_covariance_determinant_history.append(
                    np.prod(self.parent_mutation_standard_deviations, axis=1))
                self.offspring_covariance_determinant_history.append(
                    np.prod(self.offspring_mutation_standard_deviations, axis=1))
            elif self.mutation_method == "complex":
                self.parent_standard_deviation_history.append(
                    self.parent_mutation_standard_deviations)
                self.parent_covariance_determinant_history.append(
                    np.linalg.det(self.parent_covariance_matrices))
                self.offspring_covariance_determinant_history.append(
                    np.linalg.det(self.offspring_covariance_matrices))

            # *************  Check for convergence/termination      ************
            # ensure termination before 10000 iterations
            if self.objective_function_evaluation_count > self.objective_function_evaluation_max_count-self.offspring_number:
                print("max total iterations")
                break

            # termination criteria
            if max(self.parent_objectives) - min(self.parent_objectives) < self.termination_min_abs_difference:
                print("converged")
                break
            # ****************************************************************
            # *********************   Create Offspring ********************
            # I.e. perform recombination and mutation to create offspring
            self.create_new_offspring()
            # ************************************************************
            self.offspring_objective_history.append(self.offspring_objectives)
            self.offspring_x_history.append(self.offspring)

        best_x = self.parents[np.argmin(self.parent_objectives), :]
        best_objective = min(self.parent_objectives)
        return np.interp(best_x, [-1, 1], self.x_bounds), best_objective

    def initialise_random_population(self):
        self.offspring = np.random.uniform(low=-1, high=1, size=(self.offspring_number, self.x_length))
        self.offspring_objectives = np.squeeze(np.apply_along_axis(func1d=self.objective_function, arr=self.offspring, axis=1))
        if self.selection_method == "elitist":  # require pool including parents for select_parents function in this case
            self.parents = np.random.uniform(low=-1, high=1,
                                               size=(self.parent_number, self.x_length))
            self.parent_objectives = np.apply_along_axis(func1d=self.objective_function, arr=self.parents, axis=1)

    def select_parents(self):
        if self.selection_method == "standard_mew_comma_lambda":
            # choose top values in linear time
            # np.argpartition doesn't sort top values amongst themselves so is compuationally faster
            pool_objectives = self.offspring_objectives
            pool = self.offspring
            if self.mutation_method == "diagonal":
                pool_standard_deviations = self.offspring_mutation_standard_deviations
            if self.mutation_method == "complex":
                pool_standard_deviations = self.offspring_mutation_standard_deviations
                pool_rotation_matrices = self.offspring_rotation_matrices
                pool_covariance_matrices = self.offspring_covariance_matrices
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
                pool_covariance_matrices = np.zeros((self.offspring_number + self.parent_number, self.x_length, self.x_length))
                pool_covariance_matrices[0:self.offspring_number, :, :] = self.offspring_covariance_matrices
                pool_covariance_matrices[self.offspring_number:, :, :] = self.parent_covariance_matrices


        new_parent_indxs = np.argpartition(pool_objectives, self.parent_number)[:self.parent_number]
        self.parents = pool[new_parent_indxs, :]
        self.parent_objectives = pool_objectives[new_parent_indxs]

        if self.mutation_method == "diagonal":
            self.parent_mutation_standard_deviations = pool_standard_deviations[new_parent_indxs, :]
        elif self.mutation_method == "complex":
            self.parent_mutation_standard_deviations = pool_standard_deviations[new_parent_indxs, :]
            self.parent_rotation_matrices = pool_rotation_matrices[new_parent_indxs, :, :]
            self.parent_covariance_matrices = pool_covariance_matrices[new_parent_indxs, :, :]


    def create_new_offspring(self):
        """
        Recombination and Mutation
        """
        #****************   Recombination   ********************************
        # global discrete recombination for control parameters
        # global intermediate recombination for stratergy parameters
        if self.recombination_method == "global":
            # for each element in each child, inherit from a random parent
            child_recombination_indxs = np.random.choice(self.parent_number, replace=True,
                                           size = (self.offspring_number, self.x_length))
            offspring_pre_mutation = self.parents[child_recombination_indxs, np.arange(self.x_length)]
            if self.mutation_method == "diagonal" or self.mutation_method == "complex":
                child_stratergy_recombination_indxs = np.random.choice(self.parent_number, replace=True,
                                           size=(self.offspring_number,  self.x_length, 2))
                offspring_pre_mutation_standard_deviation = \
                    0.5*self.parent_mutation_standard_deviations[
                        child_stratergy_recombination_indxs[:, :, 0], np.arange(self.x_length)] +\
                    0.5*self.parent_mutation_standard_deviations[
                        child_stratergy_recombination_indxs[:, :, 1], np.arange(self.x_length)]
                if self.mutation_method == "complex":
                    child_stratergy_recombination_indxs = \
                        np.random.choice(self.parent_number, replace=True, size=(self.offspring_number*self.x_length*self.x_length, 2))
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

        #****************   Mutation   ********************************
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
                np.exp(self.mutation_tau_dash*np.broadcast_to(
                            np.random.normal(0, 1, size=(self.offspring_number, 1)), self.offspring_mutation_standard_deviations.shape)
                           + self.mutation_tau*np.random.normal(0, 1, size=self.offspring_mutation_standard_deviations.shape))
            self.offspring_mutation_standard_deviations = \
                np.clip(self.offspring_mutation_standard_deviations,
                        1e-8, self.standard_deviation_clipping_fraction_of_range*self.x_range)

            u_random_sample = np.random.normal(loc=0, scale=self.offspring_mutation_standard_deviations,
                                               size=offspring_pre_mutation.shape)
            x_new = offspring_pre_mutation + u_random_sample
            if self.bound_enforcing_method == "clipping":
                x_new = np.clip(x_new, -1, 1)
            else:
                while np.max(x_new) > 1 or np.min(x_new) < -1:
                    indxs_breaking_bounds = np.where((x_new > 1) + (x_new < -1) == 1)
                    u_random_sample = \
                        np.random.normal(loc=0, scale=self.offspring_mutation_standard_deviations
                        [indxs_breaking_bounds],size=indxs_breaking_bounds[0].size)
                    x_new[indxs_breaking_bounds] = offspring_pre_mutation[indxs_breaking_bounds] + u_random_sample

        if self.mutation_method == "complex":
            self.offspring_mutation_standard_deviations = \
                offspring_pre_mutation_standard_deviation  * \
                np.exp(self.mutation_tau_dash*np.broadcast_to(
                            np.random.normal(0, 1, size=(self.offspring_number, 1)), self.offspring_mutation_standard_deviations.shape)
                           + self.mutation_tau*np.random.normal(0, 1, size=self.offspring_mutation_standard_deviations.shape))

            self.offspring_mutation_standard_deviations = \
                np.clip(self.offspring_mutation_standard_deviations, 1e-8,
                self.standard_deviation_clipping_fraction_of_range * self.x_range)

            self.offspring_rotation_matrices = offspring_pre_mutation_rotation_matrix + self.mutation_Beta * np.random.normal(0, 1, size=(self.offspring_rotation_matrices.shape))
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
                    while np.max(self.offspring[i, :]) > 1 or np.min(self.offspring[i, :]) < -1:
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


    def update_archive(self, x_new, objective_new):
        if len(self.archive) == 0:  # if empty then initialise with the first value
            self.archive.append((x_new, objective_new))
        function_archive = [f_archive for x_archive, f_archive in self.archive]
        dissimilarity = [np.sqrt((x_archive - x_new).T @ (x_archive - x_new)) for x_archive, f_archive in self.archive]
        if min(dissimilarity) > self.archive_minimum_acceptable_dissimilarity:  # dissimilar to all points
            if len(self.archive) < self.archive_maximum_length:  # archive not full
                self.archive.append((x_new, objective_new))
            else:  # if archive is full
                if objective_new < min(function_archive):
                    self.archive[int(np.argmax(function_archive))] = (x_new, objective_new)  # replace worst solution
        else:  # new solution is close to another
            if objective_new < min(function_archive):  # objective is lowest yet
                most_similar_indx = int(np.argmin(dissimilarity))
                self.archive[most_similar_indx] = (x_new, objective_new)  # replace most similar value
            else:
                similar_and_better = np.array([dissimilarity[i] < self.archive_similar_dissimilarity and \
                                               function_archive[i] > objective_new
                                               for i in range(len(self.archive))])
                if True in similar_and_better:
                    self.archive[np.where(similar_and_better == True)[0][0]] = (x_new, objective_new)
        if self.generation_number % 10 == 0:
            # sometimes one value can like between 2 others, causing similarity even with the above loop
            # clean_archive fixes this
            # only need to do very rarely
            # slows down program a lot, so only perform when we need to visualise 2D problem
            self.clean_archive()

    def clean_archive(self):
        # first remove repeats
        for x_new, y in self.archive:
            dissimilarity = [np.sqrt((x_archive - x_new).T @ (x_archive - x_new)) for x_archive, f_archive in
                             self.archive]
            indxs_to_remove = np.where(np.array(dissimilarity) ==0)  # remove values that are close, with lower objectives
            indxs_to_remove = indxs_to_remove[0]
            if len(indxs_to_remove) > 0:
                indxs_to_remove = indxs_to_remove[1:]  # remove all but the first copy
                for i, indx_to_remove in enumerate(indxs_to_remove):
                    # deletions changes indexes so we have to adjust by i each time
                    del (self.archive[indx_to_remove - i])

        # then remove overly similar
        for x_new, y in self.archive:
            dissimilarity = [np.sqrt((x_archive - x_new).T @ (x_archive - x_new)) for x_archive, f_archive in
                             self.archive]
            indxs_to_remove = np.where((np.array(dissimilarity) < self.archive_minimum_acceptable_dissimilarity) &
                                       (self.archive_f > y))  # remove values that are close, with lower objectives
            indxs_to_remove = indxs_to_remove[0]
            if len(indxs_to_remove) > 0:
                for i, indx_to_remove in enumerate(indxs_to_remove):
                    # deletions changes indexes so we have to adjust by i each time
                    del (self.archive[indx_to_remove - i])

    def make_covariance_matrix(self):
        sigma_i = np.zeros((self.offspring_number, self.x_length, self.x_length))
        sigma_j = np.zeros((self.offspring_number, self.x_length, self.x_length))
        for offspring_number in range(self.offspring_number):
            stds = self.offspring_mutation_standard_deviations[offspring_number, :]
            sigma_i[offspring_number, np.arange(self.x_length), :] = stds
            sigma_j[offspring_number, :, np.arange(self.x_length)] = stds
        self.offspring_covariance_matrices = np.tan(2 * self.offspring_rotation_matrices) * (sigma_i**2 - sigma_j**2) * 1/2
        self.offspring_covariance_matrices = np.clip(self.offspring_covariance_matrices, -np.minimum(sigma_i, sigma_j), np.minimum(sigma_i, sigma_j))
        self.offspring_covariance_matrices[:, np.arange(self.x_length), np.arange(self.x_length)] = self.offspring_mutation_standard_deviations

    # often it was conventient to store values in lists
    # however after the optimisation it is more convenient to have
    # them as arrays, the below property methods are therefore given
    @property
    def parent_objective_history_array(self):
        return np.array(self.parent_objective_history)

    @property
    def offspring_objective_history_array(self):
        return np.array(self.offspring_objective_history)

    @property
    def parent_standard_deviation_history_array(self):
        return np.array(self.parent_standard_deviation_history)

    @property
    def parent_covariance_determinant_history_array(self):
        return np.array(self.parent_covariance_determinant_history)

    @property
    def offspring_covariance_determinant_history_array(self):
        return np.array(self.offspring_covariance_determinant_history)
    @property
    def offspring_x_history_array(self):
        return np.interp(np.array(self.offspring_x_history), [-1, 1], self.x_bounds)
    @property
    def parent_x_history_array(self):
        return np.interp(np.array(self.parent_x_history), [-1, 1], self.x_bounds)

    @property
    def archive_x(self):
        return np.interp(np.array([x_archive for x_archive, f_archive in self.archive]), [-1, 1], self.x_bounds)

    @property
    def archive_f(self):
        return np.array([f_archive for x_archive, f_archive in self.archive])



if __name__ == "__main__":
    # example run on the 5 D rana problem
    from rana import rana_func
    Comp_config = {"objective_function": rana_func,
                   "x_bounds": (-500, 500),
                   "x_length": 5,
                   "parent_number": 10,
                   "child_to_parent_ratio": 7,
                   "bound_enforcing_method": "not_clipping",
                   "selection_method": "standard_mew_comma_lambda",
                   "standard_deviation_clipping_fraction_of_range": 0.01,
                   "mutation_covariance_initialisation_fraction_of_range": 0.01,
                   "mutation_method": "complex",
                   "termination_min_abs_difference": 1e-6,
                   "maximum_archive_length": 20}
    random_seed = 1
    np.random.seed(random_seed)
    x_max = 500
    x_min = -x_max
    evo_comp = EvolutionStrategy(**Comp_config)
    x_result, objective_result = evo_comp.run()
    print(f"x_result = {x_result} \n objective_result = {objective_result}\n\n\n\
          number of objective_evaluations is {evo_comp.objective_function_evaluation_count}\
          number of generations is {evo_comp.generation_number}")
