import numpy as np

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
                 annealing_schedule="simple_exponential_cooling", halt_definition="max_function_evaluations",
                 archive_minimum_acceptable_dissimilarity=0.1,
                 maximum_markov_chain_length=10,
                 maximum_archive_length=25, step_size_initialisation_fraction_of_range=0.1,  **kwargs):

        self.x_length = x_length    # integer containing length of array x
        self.x_bounds = x_bounds    # tuple containing bounds to x
        self.x_range = 2    # when not interpolated (i.e. x bounded by -1 and 1)
        self.objective_function_raw = objective_function
        self.pertubation_method = pertubation_method
        self.annealing_schedule = annealing_schedule
        self.halt_definition = halt_definition
        self.markov_chain_maximum_length = maximum_markov_chain_length
        self.acceptances_minimum_count = round(0.6 * maximum_markov_chain_length)   # 0.6 is a heuristic from lectures

        # initialise archive and parameters determining how archive is managed
        self.archive = []   # list of (x, objective value) tuples
        self.archive_maximum_length = maximum_archive_length
        self.archive_minimum_acceptable_dissimilarity = archive_minimum_acceptable_dissimilarity
        self.archive_similar_dissimilarity = archive_minimum_acceptable_dissimilarity #*0.1 # threshold for x values being considered as similar

        self.objective_history = []
        # initialise parameters related to annealing schedule
        self.temperature_history = []
        self.Markov_chain_length = 0    # initialise to 0
        self.acceptances_count = 0  # initialise to 0
        self.acceptances_total_count = 0    # initialise to 0
        self.objective_function_evaluation_count = 0    # initialise to 0
        self.iterations_total = 0  # initialise to 0
        self.probability_of_acceptance_history = []

        if pertubation_method == "simple":  # step size just a constant in this case
            self.step_size_matrix = 2*step_size_initialisation_fraction_of_range
            # stays at initialisation in simple pertubation method
        elif pertubation_method == "Cholesky" or "Diagonal":
            self.pertubation_alpha = 0.1
            self.pertubation_omega = 2.1
            if pertubation_method == "Cholesky":
                self.step_size_control_matrix = (np.eye(x_length) *
                                                step_size_initialisation_fraction_of_range*self.x_range)**2 # squared to cancel decomposition
                # initialise, this is the matrix that gets decomposed
            else:
                self.step_size_matrix = np.eye(x_length) * \
                                                step_size_initialisation_fraction_of_range * self.x_range     # initialise, this matrix controls step size


        if annealing_schedule == "simple_exponential_cooling":
            self.annealing_alpha = 0.95   # alpha is a constant in this case
        elif annealing_schedule == "adaptive_cooling":
            self.current_temperature_accepted_objective_values=[]    # alpha calculated each iteration based on standard deviation of accepted objectives

        if halt_definition == "max_n_temperatures":
            self.temperature_maximum_iterations = kwargs['temperature_maximum_iterations']
        elif halt_definition == "max_function_evaluations":
            self.objective_function_evaluation_max_count = kwargs['maximum_function_evaluations']
        else:
            assert halt_definition == "by_improvement"
            # TODO write this
            pass

    def objective_function(self, x):
        # interpolation done here to pass the objective function x correctly interpolated
        self.objective_function_evaluation_count += 1   # increment by one everytime objective function is called
        x_interp = np.interp(x, [-1, 1], self.x_bounds)
        result = self.objective_function_raw(x_interp)
        return result


    def run(self):
        # initialise x and temperature
        x_current = self.initialise_x()
        objective_current = self.objective_function(x_current)
        self.initialise_temperature(x_current, objective_current)
        self.archive.append((x_current, objective_current))  # initialise archive
        done = False    # initialise, done = True when the optimisation has completed
        while done is False:
            x_new = self.perturb_x(x_current)
            objective_new = self.objective_function(x_new)
            self.update_archive(x_new, objective_new)
            delta_objective = objective_new - objective_current
            delta_x = x_new - x_current
            if self.accept_x_update(delta_objective, delta_x):
                # accept change if there is an improvement, or probabilisticly (based on given temperature)
                self.update_step_size(x_new, x_current)
                x_current = x_new
                objective_current = objective_new
                self.objective_history.append(objective_current)
                if self.annealing_schedule == "adaptive_cooling":
                    self.current_temperature_accepted_objective_values.append(objective_current)
                self.acceptances_count += 1 # in current markov chain
                self.acceptances_total_count += 1
            self.Markov_chain_length += 1
            self.iterations_total += 1
            done = self.temperature_scheduler()  # update temperature if need be
            x_current = self.asses_restart(x_current)

        return np.interp(x_current, [-1, 1], self.x_bounds), objective_current

    def accept_x_update(self, delta_objective, delta_x):
        if self.pertubation_method == "Diagonal":
            probability_of_accept = np.exp(-delta_objective / (self.temperature*np.sqrt(np.sum(delta_x**2))))
        else:
            probability_of_accept = np.exp(-delta_objective / self.temperature)
        self.probability_of_acceptance_history.append(probability_of_accept)
        if delta_objective < 0 or probability_of_accept > np.random.uniform(low=0, high=1):
            return True
        else:
            return False

    def initialise_x(self):
        # initialise x randomly within the given bounds
        return np.random.uniform(low=-1, high=1, size=self.x_length)

    def initialise_temperature(self, x_current, objective_current, n_steps=10, average_accept_probability=0.8):
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
            x_current = x_new
            objective_current = objective_new

        # TODO check this average increase is correct
        initial_temperature = - np.mean(objective_increase_history) / np.log(average_accept_probability)
        self.temperature = initial_temperature
        self.temperature_history.append([self.temperature, self.acceptances_total_count, self.objective_function_evaluation_count])

    def is_positive_definate(self, matrix):
        try:
            np.linalg.cholesky(matrix)
            return True
        except:
            return False

    def update_step_size(self, x_new, x_old):
        if self.pertubation_method == "simple":
            return
        elif self.pertubation_method == "Cholesky":
            clipping_fraction_of_range = 0.5
            covariance = np.cov([x_new, x_old], rowvar=False)
            self.step_size_control_matrix = (1 - self.pertubation_alpha) * self.step_size_control_matrix + \
                                            self.pertubation_alpha * self.pertubation_omega * covariance
            self.step_size_control_matrix = np.clip(self.step_size_control_matrix,
                                                    -self.x_range*clipping_fraction_of_range,
                                                    self.x_range*clipping_fraction_of_range)
            i = 0
            while not self.is_positive_definate(self.step_size_control_matrix): #np.min(np.linalg.eigvals(self.step_size_control_matrix)) > 1e-16:
                i += 1
                self.step_size_control_matrix += np.eye(self.x_length)*1e-16*1000**i  # to make positive definate
                if i > 3:
                    raise Exception("couldn't get positive definate step size control matrix")

        elif self.pertubation_method == "Diagonal":
            clipping_fraction_of_range = 0.5
            self.step_size_matrix = (1-self.pertubation_alpha)*self.step_size_matrix + \
                                   np.diag(self.pertubation_alpha*self.pertubation_omega*np.abs(x_new - x_old))
            self.step_size_matrix = np.clip(self.step_size_matrix, self.x_range*1e-16,
                                            self.x_range*clipping_fraction_of_range)  # clip stepsize to not be too large
            # noise sampled in both directions so can clip step size matrix to be bounded by just over 0
            # instead of big negative and big positve numbers - this prevents any step size from going to 0 or becoming
            # too large

    def perturb_x(self, x):
        if self.pertubation_method == "simple":
            u_random_sample = np.random.uniform(low=-1, high=1, size=self.x_length)
            x_new = x + self.step_size_matrix * u_random_sample  # constant step size

        elif self.pertubation_method == "Cholesky":
            u_random_sample = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=self.x_length)
            Q = np.linalg.cholesky(self.step_size_control_matrix)
            x_new = x + Q@u_random_sample

        elif self.pertubation_method == "Diagonal":
            u_random_sample = np.random.uniform(low=-1, high=1, size=self.x_length)
            x_new = x+self.step_size_matrix@u_random_sample

        # if max(x_new) > 1 or min(x_new) < -1:
        #     x_new = self.perturb_x(x)   # recursively call perturb until sampled within bounds
        #     return x_new
        # else:
        #     return x_new
        return np.clip(x_new, -1, 1)

    def asses_restart(self, x, min_difference = 0.01):
        # lets assume roughly to asses restart over the length of half a markov chain max length
        if len(self.objective_history) % int(self.markov_chain_maximum_length/2) == 0 and \
            len(self.objective_history) > self.markov_chain_maximum_length:
            if max(self.objective_history[-50:]) - min(self.objective_history[-50:]) < min_difference:
                # then rebase from best archive solution
                x_restart = self.archive_x[np.argmax(self.archive_f), :]
                print("restarted")
                return x_restart
        else:
            return x


    def temperature_scheduler(self):
        if self.Markov_chain_length > self.markov_chain_maximum_length or \
                self.acceptances_count > self.acceptances_minimum_count:
            if self.annealing_schedule == "simple_exponential_cooling":
                self.temperature = self.temperature * self.annealing_alpha
            elif self.annealing_schedule == "adaptive_cooling":
                if len(self.current_temperature_accepted_objective_values) == 1:
                    self.alpha = 0.5
                else:
                    latest_temperature_standard_dev = np.std(self.current_temperature_accepted_objective_values)
                    self.alpha = np.max([0.5, np.exp(-0.7*self.temperature/latest_temperature_standard_dev)])

                self.temperature = self.temperature * self.alpha
                self.current_temperature_accepted_objective_values = []     # reset
            self.temperature_history.append([self.temperature, self.acceptances_total_count, self.objective_function_evaluation_count])
            done = self.get_halt()
            self.Markov_chain_length = 0    # restart counter
            self.acceptances_count = 0  # restart counter
        else:   # no temperature change
            done = False
        return done

    def get_halt(self):
        if self.halt_definition == "max_n_temperatures":
            if len(self.temperature_history) > self.temperature_maximum_iterations:
                done = True  # stopping criteria has been met
            else:
                done = False
        elif self.halt_definition == "max_function_evaluations":
            if self.objective_function_evaluation_count > self.objective_function_evaluation_max_count:
                done = True
            else:
                done = False
        return done

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
    def objective_history_array(self):
        return self.objective_history





if __name__ == "__main__":
    np.random.seed(0)
    test = "Cholesky"   # "simple_rana" "adaptive_cooling"  "Cholesky"

    if test == "Diagonal":
        from rana import rana_func
        x_max = 500
        x_min = -x_max
        rana_2d = SimulatedAnnealing(x_length=3, x_bounds=(x_min, x_max), objective_function=rana_func,
                                     annealing_schedule="adaptive_cooling", pertubation_method="Diagonal",
                                     maximum_archive_length=100,
                                     archive_minimum_acceptable_dissimilarity=60, maximum_markov_chain_length=50,
                                     temperature_maximum_iterations=200, pertubation_fraction_of_range=0.1)
        x_result, objective_result = rana_2d.run()
        print(f"x_result = {x_result} \n objective_result = {objective_result} \n "
              f"number of function evaluations = {rana_2d.objective_function_evaluation_count}")

        archive_x = np.array([x_archive for x_archive, f_archive in rana_2d.archive])
        archive_f = np.array([f_archive for x_archive, f_archive in rana_2d.archive])

    if test == "Cholesky":
        from rana import rana_func

        random_seed = 0
        maximum_markov_chain_length = 50
        x_length = 5
        np.random.seed(random_seed)
        np.random.seed(random_seed)
        x_max = 500
        x_min = -x_max
        rana_2d_chol = SimulatedAnnealing(x_length=x_length, x_bounds=(x_min, x_max), objective_function=rana_func,
                                          pertubation_method="Cholesky", maximum_archive_length=100,
                                          maximum_markov_chain_length=maximum_markov_chain_length,
                                          maximum_function_evaluations=10000)
        x_result_chol, objective_result_chol = rana_2d_chol.run()
        print(f"x_result = {x_result_chol} \n objective_result = {objective_result_chol} \n "
              f"number of function evaluations = {rana_2d_chol.objective_function_evaluation_count}")

    if test == "adaptive_cooling":
        from rana import rana_func
        x_max = 500
        x_min = -x_max
        rana_2d = SimulatedAnnealing(x_length=2, x_bounds=(x_min, x_max), objective_function=rana_func,
                                     annealing_schedule="adaptive_cooling",
                                     maximum_archive_length=100,
                                     archive_minimum_acceptable_dissimilarity=60, maximum_markov_chain_length=50,
                                     temperature_maximum_iterations=200, pertubation_fraction_of_range=0.1)
        x_result, objective_result = rana_2d.run()
        print(f"x_result = {x_result} \n objective_result = {objective_result} \n "
              f"number of function evaluations = {rana_2d.objective_function_evaluation_count}")

        archive_x = np.array([x_archive for x_archive, f_archive in rana_2d.archive])
        archive_f = np.array([f_archive for x_archive, f_archive in rana_2d.archive])

    if test == "simple_rana":  # rana function, simple Simulated Annealing config
        from rana import rana_func
        x_max = 500
        x_min = -x_max
        rana_2d = SimulatedAnnealing(x_length=2, x_bounds=(x_min, x_max), objective_function=rana_func,
                                     maximum_archive_length=50,
                                     archive_minimum_acceptable_dissimilarity=20, maximum_markov_chain_length=50,
                                     temperature_maximum_iterations=200, pertubation_fraction_of_range=0.1)
        x_result, objective_result = rana_2d.run()
        print(f"x_result = {x_result} \n objective_result = {objective_result} \n "
              f"number of function evaluations = {rana_2d.objective_function_evaluation_count}")

        archive_x = np.array([x_archive for x_archive, f_archive in rana_2d.archive])
        archive_f = np.array([f_archive for x_archive, f_archive in rana_2d.archive])

    if test == 0:   # simplest objective
        x_max = 50
        x_min = -x_max
        simple_objective = lambda x: x + np.sin(x)*20 + 3
        simple_anneal = SimulatedAnnealing(x_length=1, x_bounds=(x_min, x_max), objective_function=simple_objective,
                                           archive_minimum_acceptable_dissimilarity=5, maximum_markov_chain_length=50,
                                           temperature_maximum_iterations=500, pertubation_fraction_of_range=0.1)
        x_result, objective_result = simple_anneal.run()
        print(f"x_result = {x_result} \n objective_result = {objective_result}")

        archive_x = np.array([x_archive for x_archive, f_archive in simple_anneal.archive])
        archive_f = np.array([f_archive for x_archive, f_archive in simple_anneal.archive])

        import matplotlib.pyplot as plt
        x_linspace = np.linspace(x_min, x_max, 200)
        plt.plot(x_linspace, simple_objective(x_linspace))
        plt.plot(x_result, objective_result, "or")
        plt.plot(archive_x, archive_f, "xr")
        plt.show()

        plt.plot(simple_anneal.objective_history)
        plt.show()

    if test == 1:
        simple_objective = lambda x: x[0]**2 + np.sin(x[1])
        simple_anneal = SimulatedAnnealing(x_length=2, x_bounds=(-10, 10), objective_function=simple_objective,
                                           archive_minimum_acceptable_dissimilarity=3)
        x_result, objective_result = simple_anneal.run()
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



