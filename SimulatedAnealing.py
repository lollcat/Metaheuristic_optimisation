import numpy as np
from collections import deque

class SimulatedAnnealing:
    """
    Currently assumes:
        - interpolation is done inside objective_function wrapper function, elsewhere in the class x is
        bounded between -1 and 1
        - elements of x are all on the same scale
    Notes:
        Variable and function names written such that most of the code should be self-explanatory
    """
    def __init__(self, x_length, x_bounds, objective_function,
                 pertubation_method="simple",
                 annealing_schedule="simple_exponential_cooling", with_restarts=False,
                 archive_minimum_acceptable_dissimilarity=0.2,
                 maximum_markov_chain_length=10, bound_enforcing_method="clipping",
                 maximum_archive_length=20, step_size_initialisation_fraction_of_range=0.5, annealing_alpha=0.95,
                 maximum_function_evaluations=10000,
                 cholesky_path_length=5,
                 pertubation_alpha = 0.1, pertubation_omega = 2.1,
                 convergence_min_improvement = 1e-6,
                 update_step_size_every_step = True,
                 minimum_determinant = 1e-16,
                 ):

        self.x_length = x_length    # integer containing length of array x
        self.x_bounds = x_bounds    # tuple containing bounds to x
        self.x_range = 2    # when not interpolated (i.e. x bounded by -1 and 1)
        self.objective_function_raw = objective_function
        self.pertubation_method = pertubation_method
        self.annealing_schedule = annealing_schedule
        self.markov_chain_maximum_length = maximum_markov_chain_length
        self.acceptances_minimum_count = round(0.6 * maximum_markov_chain_length)   # 0.6 is a heuristic from lectures
        self.convergence_min_improvement = convergence_min_improvement
        self.convergence_evaluation_window = int(maximum_function_evaluations/10)
        self.update_step_size_every_step = update_step_size_every_step

        # initialise archive and parameters determining how archive is managed
        self.archive = []   # list of (x, objective value) tuples
        self.archive_maximum_length = maximum_archive_length
        self.archive_minimum_acceptable_dissimilarity = archive_minimum_acceptable_dissimilarity
        self.archive_similar_dissimilarity = archive_minimum_acceptable_dissimilarity #*0.1 # threshold for x values being considered as similar

        self.accepted_objective_history = []
        self.acceptences_locations = []
        self.accepted_x_history = []
        self.objective_history = []
        self.x_history = []
        self.alpha_history = []
        self.iterations_without_acceptance = 0
        self.bound_enforcing_method = bound_enforcing_method
        # initialise parameters related to annealing schedule
        self.temperature_history = []
        self.step_size_matrix_history = []
        self.Markov_chain_length = 0    # initialise to 0
        self.acceptances_count = 0  # initialise to 0
        self.acceptances_total_count = 0    # initialise to 0
        self.objective_function_evaluation_count = 0    # initialise to 0
        self.iterations_total = 0  # initialise to 0
        self.probability_of_acceptance_history = []
        self.restart_count = 0
        self.max_iterations_without_acceptance_till_restart = int(maximum_function_evaluations/10)
        self.with_restarts = with_restarts

        if pertubation_method == "simple":  # step size just a constant in this case
            self.step_size_matrix = 2*step_size_initialisation_fraction_of_range
            # stays at initialisation in simple pertubation method
        elif pertubation_method == "Cholesky" or "Diagonal":
            self.pertubation_alpha = pertubation_alpha
            self.pertubation_omega = pertubation_omega
            self.recent_x_history = deque(maxlen=cholesky_path_length)#self.markov_chain_maximum_length)
            if pertubation_method == "Cholesky":
                self.step_size_control_matrix = (np.eye(x_length) *
                                                step_size_initialisation_fraction_of_range*self.x_range)**2 # squared to cancel out sqrting
                self.step_size_matrix = np.linalg.cholesky(self.step_size_control_matrix)
            else:
                self.step_size_matrix = np.eye(x_length) * \
                                                step_size_initialisation_fraction_of_range * self.x_range     # initialise, this matrix controls step size


        if annealing_schedule == "simple_exponential_cooling":
            self.annealing_alpha = annealing_alpha   # alpha is a constant in this case
        elif annealing_schedule == "adaptive_cooling":
            self.current_temperature_accepted_objective_values = []   # alpha calculated off standard deviation of this

        self.objective_function_evaluation_max_count = maximum_function_evaluations


    def objective_function(self, x):
        # interpolation done here to pass the objective function x correctly interpolated
        self.objective_function_evaluation_count += 1   # increment by one everytime objective function is called
        x_interp = np.interp(x, [-1, 1], self.x_bounds)
        result = self.objective_function_raw(x_interp)
        return result


    def run(self):
        # initialise x and temperature
        self.x_current = self.initialise_x()
        self.objective_current = self.objective_function(self.x_current)
        self.initialise_temperature(self.x_current, self.objective_current)
        self.archive.append((self.x_current, self.objective_current))  # initialise archive
        if self.pertubation_method == "Cholesky":
            self.recent_x_history.append(self.x_current)
        done = False    # initialise, done = True when the optimisation has completed
        while done is False:
            x_new = self.perturb_x(self.x_current)
            objective_new = self.objective_function(x_new)
            delta_objective = objective_new - self.objective_current
            delta_x = x_new - self.x_current
            self.objective_history.append(objective_new)
            self.x_history.append(x_new)
            if self.accept_x_update(delta_objective, delta_x):
                self.update_archive(x_new, objective_new)
                # accept change if there is an improvement, or probabilisticly (based on given temperature)
                if self.pertubation_method == "Cholesky":   # store recent x values to get covariance matrix
                    self.recent_x_history.append(x_new)
                elif self.pertubation_method == "Diagonal":
                    self.latest_accepted_step = x_new - self.x_current

                self.x_current = x_new
                self.objective_current = objective_new
                if self.annealing_schedule == "adaptive_cooling":
                    # record accepted objective values in the current chain
                    self.current_temperature_accepted_objective_values.append(self.objective_current)
                self.accepted_objective_history.append([objective_new, self.objective_function_evaluation_count])
                self.acceptences_locations.append(self.objective_function_evaluation_count)  # when in optimisation acceptence occured
                self.accepted_x_history.append(x_new)
                self.acceptances_count += 1 # in current markov chain
                self.acceptances_total_count += 1
                self.iterations_without_acceptance = 0
                if not self.update_step_size_every_step: # TODO see what is said in email
                    self.update_step_size()
            else:
                self.iterations_without_acceptance += 1
            self.Markov_chain_length += 1
            self.iterations_total += 1
            done = self.temperature_scheduler()  # update temperature if need be
            if self.update_step_size_every_step:
                self.update_step_size()
            if self.with_restarts:
                self.asses_restart()
            if self.restart_count > 5:
                print("max restarts reached, stopping early")
                break

        return np.interp(self.x_current, [-1, 1], self.x_bounds), self.objective_current

    def accept_x_update(self, delta_objective, delta_x):
        if delta_objective < 0:
            #self.probability_of_acceptance_history.append([1, self.objective_function_evaluation_count])
            return True
        else:
            if self.pertubation_method == "Diagonal":
                probability_of_accept = np.exp(-delta_objective / (self.temperature*np.sqrt(np.sum(delta_x**2))))
            else:
                probability_of_accept = np.exp(-delta_objective / self.temperature)
            self.probability_of_acceptance_history.append([np.clip(probability_of_accept, 0, 1), self.objective_function_evaluation_count])
            if probability_of_accept > np.random.uniform(low=0, high=1):
                return True
            else:
                return False

    def initialise_x(self):
        # initialise x randomly within the given bounds
        return np.random.uniform(low=-1, high=1, size=self.x_length)

    def initialise_temperature(self, x_current, objective_current, n_steps=60, average_accept_probability=0.8):
        """
        Initialises system temperature
        As all x's are initially accepted, x does a random walk, so changes in x are not returned
        """
        objective_increase_history = []  # if many samples are taken then this could be changed to running average
        for step in range(1, n_steps+1):
            x_new = self.perturb_x(x_current)
            objective_new = self.objective_function(x_new)
            if objective_new > objective_current:
                objective_increase_history.append(objective_new - objective_current)
            if step == n_steps:
                self.latest_accepted_step = x_new - self.x_current
            x_current = x_new
            objective_current = objective_new

        initial_temperature = - np.mean(objective_increase_history) / np.log(average_accept_probability)
        self.temperature = initial_temperature
        self.temperature_history.append([self.temperature, self.acceptances_total_count, self.objective_function_evaluation_count])

    def is_positive_definate(self, matrix):
        try:
            np.linalg.cholesky(matrix)
            return True
        except:
            return False

    def update_step_size(self):
        if self.pertubation_method == "simple":
            return
        elif self.pertubation_method == "Cholesky":
            #covariance = np.cov([x_new, x_old], rowvar=False)
            covariance = np.cov(self.recent_x_history, rowvar=False)
            covariance = np.clip(covariance, -self.x_range/2, self.x_range/2)   # prevent covariance from becoming too large
            self.step_size_control_matrix = (1 - self.pertubation_alpha) * self.step_size_control_matrix + \
                                            self.pertubation_alpha * self.pertubation_omega * covariance
            # conservative clipping
            self.step_size_control_matrix = self.step_size_control_matrix.clip(-self.x_range*2, self.x_range*2)
            i = 0
            while not self.is_positive_definate(self.step_size_control_matrix):
                i += 1
                self.step_size_control_matrix += np.eye(self.x_length)*1e-6*10**i  # to make positive definate
                if i > 7:
                    raise Exception("couldn't get positive definate step size control matrix")
            if np.linalg.det(self.step_size_matrix) < 1e-16: # double step size if determinant falls below 1e-6
                self.step_size_control_matrix = self.step_size_control_matrix * 1.1
                # now have to enforce positive definateness again
                while not self.is_positive_definate(self.step_size_control_matrix):
                    i += 1
                    self.step_size_control_matrix += np.eye(self.x_length) * 1e-6 * 10 ** i
                    if i > 7:
                        raise Exception("couldn't get positive definate step size control matrix")
            self.step_size_matrix = np.linalg.cholesky(self.step_size_control_matrix)
            self.step_size_matrix_history.append(self.step_size_matrix)

        elif self.pertubation_method == "Diagonal":

            self.step_size_matrix = (1-self.pertubation_alpha)*self.step_size_matrix + \
                                   np.diag(self.pertubation_alpha*self.pertubation_omega*np.abs(self.latest_accepted_step))
            # conservative clipping
            self.step_size_matrix = np.clip(self.step_size_matrix, self.x_range * 1e-16, self.x_range*2)

            if np.linalg.det(self.step_size_matrix) < 1e-16:  # double step size if determinant falls below 1e-6
                self.step_size_matrix = self.step_size_matrix*2
            self.step_size_matrix_history.append(np.diag(self.step_size_matrix))

    def perturb_x(self, x):
        if self.pertubation_method == "simple":
            u_random_sample = np.random.uniform(low=-1, high=1, size=self.x_length)
            x_new = x + self.step_size_matrix * u_random_sample  # constant step size
            if self.bound_enforcing_method == "clipping":
                return np.clip(x_new, -1, 1)
            else:
                while max(x_new) > 1 or min(x_new) < -1:
                    indxs_breaking_bounds = np.where((x_new > 1) + (x_new < -1) == 1)
                    u_random_sample = np.random.uniform(low=-1, high=1, size=indxs_breaking_bounds[0].size)
                    x_new[indxs_breaking_bounds] = x[indxs_breaking_bounds] + self.step_size_matrix * u_random_sample  # constant step size


        elif self.pertubation_method == "Cholesky":
            u_random_sample = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=self.x_length)
            x_new = x + self.step_size_matrix@u_random_sample
            if self.bound_enforcing_method == "clipping":
                return np.clip(x_new, -1, 1)
            else:
                if max(x_new) > 1 or min(x_new) < -1:
                     x_new = self.perturb_x(x)   # recursively call perturb until sampled within bounds

        elif self.pertubation_method == "Diagonal":
            u_random_sample = np.random.uniform(low=-1, high=1, size=self.x_length)
            x_new = x+self.step_size_matrix@u_random_sample
            if self.bound_enforcing_method == "clipping":
                return np.clip(x_new, -1, 1)
            else:
                while max(x_new) > 1 or min(x_new) < -1:
                     #x_new = self.perturb_x(x)   # recursively call perturb until sampled within bounds
                    indxs_breaking_bounds = np.where((x_new > 1) + (x_new < -1) == 1)
                    u_random_sample = np.random.uniform(low=-1, high=1, size=indxs_breaking_bounds[0].size)
                    x_new[indxs_breaking_bounds] = x[indxs_breaking_bounds] + np.diag(np.diag(self.step_size_matrix)[indxs_breaking_bounds])@u_random_sample

        return x_new


    def asses_restart(self, min_difference = 0.01):
        length = self.markov_chain_maximum_length  # base on length of markov chain
        if len(self.objective_history) % length == 0 and \
            len(self.accepted_objective_history) > self.markov_chain_maximum_length:
            if max(self.accepted_objective_history_array[-length:, 0]) - min(self.accepted_objective_history_array[-length:, 0]) < min_difference:
                # then rebase from best archive solution
                x_restart = self.archive_x[np.argmax(self.archive_f), :]
                print("restarted due to minimal progress")
                self.restart_count += 1
                self.x_current = x_restart
            elif self.iterations_without_acceptance > self.max_iterations_without_acceptance_till_restart:
                x_restart = self.archive_x[np.argmax(self.archive_f), :]
                print(f"restarted due to {self.max_iterations_without_acceptance_till_restart} iterations "
                      f"without acceptence")
                self.restart_count += 1
                self.x_current = x_restart


    def temperature_scheduler(self):
        if self.Markov_chain_length > self.markov_chain_maximum_length or \
                self.acceptances_count > self.acceptances_minimum_count:
            #if len(self.current_temperature_accepted_objective_values) <= 1:
             #   x_restart = self.archive_x[np.argmax(self.archive_f), :]
             #   self.x_current = x_restart
             #   print("restarted due to no accepted across chain")
             #   self.restart_count += 1
            if self.annealing_schedule == "simple_exponential_cooling":
                self.temperature = self.temperature * self.annealing_alpha
            elif self.annealing_schedule == "adaptive_cooling":
                #raise Exception("incomplete code")

                multiplier = (0.05 / 0.8) ** (1 / 10000)
                desired_probability = 0.9*multiplier**self.objective_function_evaluation_count
                length = int(self.markov_chain_maximum_length/2)
                if self.probability_of_acceptance_history_array.shape[0] >  length:
                    if np.mean(self.probability_of_acceptance_history_array[-length:, 0]) < desired_probability:
                        self.alpha = 1.5
                    else:
                        self.alpha = 0.8
                else:
                    self.alpha = 0.95
                self.alpha_history.append([self.alpha, self.objective_function_evaluation_count])

                """
                if len(self.current_temperature_accepted_objective_values) <= 1:
                    self.alpha = 2
                    self.alpha_history.append([self.alpha, self.objective_function_evaluation_count])
                else:
                    latest_temperature_standard_dev = np.std(self.current_temperature_accepted_objective_values)
                    self.alpha = np.max([0.5, np.exp(-0.7*self.temperature/latest_temperature_standard_dev)])
                    self.alpha_history.append([self.alpha, self.objective_function_evaluation_count])
                """
                self.temperature = self.temperature * self.alpha
            if np.isnan(self.temperature):
                raise Exception("temperature is nan")
            self.current_temperature_accepted_objective_values = []     # reset
            self.temperature_history.append([self.temperature, self.acceptances_total_count, self.objective_function_evaluation_count])

            self.Markov_chain_length = 0    # restart counter
            self.acceptances_count = 0  # restart counter
        done = self.get_halt()
        if done is True:
            self.temperature_history.append(
                [self.temperature, self.acceptances_total_count, self.objective_function_evaluation_count])
            self.accepted_objective_history.append([self.objective_current, self.objective_function_evaluation_count])
            self.acceptences_locations.append(self.objective_function_evaluation_count) # when in optimisation acceptence occured
            self.accepted_x_history.append(self.x_current)
        return done

    def get_halt(self):
        """
        1.  first check convergence, converge if over the last evaluation window (5% of total max function evals), the
         diffrence between the maximum and minimum accepted values (within the window) is below the threshold defined
         by self.convergence_min_improvement (typically set to 1e-8)
        2. If the maximum number of function evaluations has been reached (typically set to 10 000) then end program
        """
        if self.objective_function_evaluation_count % self.convergence_evaluation_window == 0:
            # only make this check every self.convergence_evaluation_window number of iterations
            acceptences_locations_array = np.array(self.acceptences_locations)
            acceptence_indx_over_window = \
            np.arange(len(acceptences_locations_array))[
                          acceptences_locations_array >
                          self.objective_function_evaluation_count - self.convergence_evaluation_window]
            if len(acceptence_indx_over_window) < 2: # 1 or 0 acceptences within last window implies convergence
                print("converged")
                done = True
                return done
            else:  # caclulate diffrence between max and min over window
                earliest_acceptence_indx_over_window = acceptence_indx_over_window[0]
                best_accepted_value_over_recent_window = \
                    np.max(self.accepted_objective_history_array[earliest_acceptence_indx_over_window:, 0])
                worst_accepted_value_over_recent_window = \
                    np.min(self.accepted_objective_history_array[earliest_acceptence_indx_over_window:, 0])

                if best_accepted_value_over_recent_window - worst_accepted_value_over_recent_window < \
                        self.convergence_min_improvement:
                    print("converged")
                    done = True
                    return done
        # check if max iter has been reached
        if self.objective_function_evaluation_count >= self.objective_function_evaluation_max_count:
            done = True
        else:
            done = False
        return done

    def update_archive(self, x_new, objective_new):
        function_archive = [f_archive for x_archive, f_archive in self.archive]
        dissimilarity = [np.sqrt((x_archive - x_new).T @ (x_archive - x_new)) for x_archive, f_archive in self.archive]
        if min(dissimilarity) > self.archive_minimum_acceptable_dissimilarity:  # dissimilar to all points
            if len(self.archive) < self.archive_maximum_length:  # archive not full
                self.archive.append((x_new, objective_new))
            else:  # if archive is full
                if objective_new < min(function_archive):
                    self.archive[int(np.argmax(function_archive))] = (x_new, objective_new)  # replace worst solution
        else:    # new solution is close to another
            if objective_new < min(function_archive):   # objective is lowest yet
                most_similar_indx = int(np.argmin(dissimilarity))
                self.archive[most_similar_indx] = (x_new, objective_new)  # replace most similar value
            else:
                similar_and_better = np.array([dissimilarity[i] < self.archive_similar_dissimilarity and \
                                      function_archive[i] > objective_new
                                      for i in range(len(self.archive))])
                if True in similar_and_better:
                    self.archive[np.where(similar_and_better == True)[0][0]] = (x_new, objective_new)
        if self.objective_function_evaluation_count % (int(self.objective_function_evaluation_max_count/10)) == 0:
            # sometimes one value can like between 2 others, causing similarity even with the above loop
            # clean_archive fixes this
            # only need to do very rarely
            self.clean_archive()


    def clean_archive(self):
        for x_new, y in self.archive:
            dissimilarity = [np.sqrt((x_archive - x_new).T @ (x_archive - x_new)) for x_archive, f_archive in
                             self.archive]
            indxs_to_remove = np.where((np.array(dissimilarity) < self.archive_minimum_acceptable_dissimilarity) &
                                       (self.archive_f > y))  # remove values that are close, with lower objectives
            indxs_to_remove = indxs_to_remove[0]
            if len(indxs_to_remove) > 0:
                for i, indx_to_remove in enumerate(indxs_to_remove):
                    # deletions changes indexes so we have to adjust by i each time
                    del(self.archive[indx_to_remove - i])



    @property
    def temperature_history_array(self):
        return np.array(self.temperature_history)

    @property
    def archive_x(self):
        return np.interp(np.array([x_archive for x_archive, f_archive in self.archive]), [-1, 1], self.x_bounds)

    @property
    def archive_f(self):
        return np.array([f_archive for x_archive, f_archive in self.archive])

    @property
    def accepted_objective_history_array(self):
        return np.array(self.accepted_objective_history)

    @property
    def accepted_x_history_array(self):
        return np.interp(np.array(self.accepted_x_history), [-1, 1], self.x_bounds)

    @property
    def objective_history_array(self):
        return np.array(self.objective_history)

    @property
    def step_size_matrix_history_array(self):
        return np.array(self.step_size_matrix_history)

    @property
    def probability_of_acceptance_history_array(self):
        return np.array(self.probability_of_acceptance_history)

    @property
    def x_history_array(self):
        return np.interp(np.array(self.x_history), [-1, 1], self.x_bounds)

    @property
    def alpha_history_array(self):
        return np.array(self.alpha_history)

    @property
    def eigenvalue_eigenvector_history(self):
        theta_history = []
        eigen_values_history = []
        for i in range(self.step_size_matrix_history_array.shape[0]):
            step_size_matrix = self.step_size_matrix_history_array[i, :, :]
            eigenvalues, eigenvectors = np.linalg.eig(step_size_matrix)
            thetas = np.arccos(np.eye(self.x_length) @ eigenvectors)
            min_thetas = np.min(thetas, axis=0)
            order = np.argsort(-eigenvalues)
            eigen_values_history.append(list(eigenvalues[order]))
            theta_history.append(list(min_thetas[order]))
        return np.array(eigen_values_history), np.array(theta_history)






if __name__ == "__main__":
    np.random.seed(0)

    from rana import rana_func

    configuration = {"pertubation_method": "simple",
                     "x_length": 2,
                     "x_bounds": (-500, 500),
                     "annealing_schedule": "simple_exponential_cooling",
                     "objective_function": rana_func,
                     "maximum_archive_length": 100,
                     "archive_minimum_acceptable_dissimilarity": 0.2,
                     "maximum_markov_chain_length": 50,
                     "maximum_function_evaluations": 10000,
                     "step_size_initialisation_fraction_of_range": 0.1,
                     "bound_enforcing_method": "not_clipping",
                     "cholesky_path_length": 5,
                     }
    np.random.seed(3)
    rana_2d_chol = SimulatedAnnealing(**configuration)
    x_result_chol, objective_result_chol = rana_2d_chol.run()
    print(f"x_result = {x_result_chol} \n objective_result = {objective_result_chol} \n "
          f"number of function evaluations = {rana_2d_chol.objective_function_evaluation_count}")
    print(f"best objective result {rana_2d_chol.objective_history_array.min()}")
