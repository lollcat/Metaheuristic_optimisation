import numpy as np

class SimulatedAnnealing:
    """
    Currently assumes:
        - elements of x are all on the same scale
    Notes:
        Variable and function names written such that most of the code should be self-explanatory
    """
    def __init__(self, x_length, x_bounds, objective_function, pertubation_method="simple",
                 annealing_schedule="simple_exponential_cooling", halt_definition="max_n_temperatures",
                 maximum_markov_chain_length=10,
                 maximum_archive_length=25, archive_minimum_acceptable_dissimilarity=0.1, **kwargs):

        self.x_length = x_length    # integer containing length of array x
        self.x_bounds = x_bounds    # tuple containing bounds to x
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

        # initialise parameters related to annealing schedule
        self.temperature_history = []
        self.Markov_chain_length = 0    # initialise to 0
        self.acceptances_count = 0  # initialise to 0
        self.objective_function_evaluation_count = 0    # initialise to 0

        if pertubation_method == "simple":
            self.pertubation_fraction_of_range = kwargs['pertubation_fraction_of_range']

        if annealing_schedule == "simple_exponential_cooling":
            self.alpha = 0.95   # alpha is a constant in this case

        if halt_definition == "max_n_temperatures":
            self.temperature_maximum_iterations = kwargs['temperature_maximum_iterations']
        elif halt_definition == "max_function_evaluations":
            self.objective_function_evaluation_max_count = kwargs['maximum_function_evaluations']
        else:
            assert halt_definition == "by_improvement"
            # TODO write this
            pass

    def objective_function(self, *args, **kwargs):
        self.objective_function_evaluation_count += 1   # increment by one everytime objective function is called
        return self.objective_function_raw(*args, **kwargs)


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
            if delta_objective < 0 or np.exp(-delta_objective/self.temperature) > np.random.uniform(low=0, high=1):
                # accept change if there is an improvement, or probabilisticly (based on given temperature)
                x_current = x_new
                objective_current = objective_new
                self.acceptances_count += 1
            self.Markov_chain_length += 1
            done = self.temperature_scheduler()  # update temperature if need be

        return x_current, objective_current




    def initialise_x(self):
        # initialise x randomly within the given bounds
        return np.random.uniform(low=self.x_bounds[0], high=self.x_bounds[1], size=self.x_length)

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
        self.temperature_history.append(initial_temperature)


    def perturb_x(self, x):
        if self.pertubation_method == "simple":
            D_max_change = (self.x_bounds[1] - self.x_bounds[0]) * self.pertubation_fraction_of_range
            u_random_sample = np.random.uniform(low=-1, high=1, size=self.x_length)
            x_new = x + u_random_sample * D_max_change, self.x_bounds[0], self.x_bounds[1]
            return np.clip(x + u_random_sample*D_max_change, self.x_bounds[0], self.x_bounds[1])

            # if max(x_new) > self.x_bounds[1] or min(x_new) < self.x_bounds[0]:
            #     x_new = self.perturb_x(x)   # recursively call perturb until sampled within bounds
            #     return x_new
            # else:
            #     return x_new

    def temperature_scheduler(self):
        if self.Markov_chain_length > self.markov_chain_maximum_length or \
                self.acceptances_count > self.acceptances_minimum_count:
            self.Markov_chain_length = 0    # restart counter
            self.acceptances_count = 0  # restart counter
            if self.annealing_schedule == "simple_exponential_cooling":
                self.temperature = self.temperature * self.alpha
                self.temperature_history.append(self.temperature)
            done = self.get_halt()
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
        if max(dissimilarity) > self.archive_minimum_acceptable_dissimilarity:
            if len(self.archive) < self.archive_maximum_length:  # archive not full
                self.archive.append((x_new, objective_new))
            else:  # if archive is full
                if objective_new < min(function_archive):
                    self.archive[int(np.argmax(function_archive))] = (x_new, objective_new)  # replace worst solution
        else:    # new solution is close to another
            if objective_new < max(function_archive):
                self.archive[int(np.argmin(dissimilarity))] = (x_new, objective_new)  # replace most similar value
            else:
                similar_and_better = np.array([dissimilarity[i] < self.archive_similar_dissimilarity and \
                                      function_archive[i] > objective_new
                                      for i in range(len(self.archive))])
                if True in similar_and_better:
                    self.archive[np.where(similar_and_better == True)[0]] = (x_new, objective_new)



if __name__ == "__main__":
    np.random.seed(2)
    test = "rana"
    if test == "rana":  # rana function
        from rana import rana_func
        x_max = 500
        x_min = -x_max
        rana_2d = SimulatedAnnealing(x_length=2, x_bounds=(x_min, x_max), objective_function=rana_func,
                                           archive_minimum_acceptable_dissimilarity=1, maximum_markov_chain_length=100,
                                           temperature_maximum_iterations=50, pertubation_fraction_of_range=0.1)
        x_result, objective_result = rana_2d.run()
        print(f"x_result = {x_result} \n objective_result = {objective_result} \n "
              f"number of function evaluations = {rana_2d.objective_function_evaluation_count}")

        archive_x = np.array([x_archive for x_archive, f_archive in rana_2d.archive])
        archive_f = np.array([f_archive for x_archive, f_archive in rana_2d.archive])


    if test == 0:  # simplest objective
        x_max = 50
        x_min = -x_max
        simple_objective = lambda x: x + np.sin(x)*20 + 3
        simple_anneal = SimulatedAnnealing(x_length=1, x_bounds=(x_min, x_max), objective_function=simple_objective,
                                           archive_minimum_acceptable_dissimilarity=1, maximum_markov_chain_length=50,
                                           temperature_maximum_iterations=100, pertubation_fraction_of_range=0.1)
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

    if test == 1:
        simple_objective = lambda x: x[0]**2 + np.sin(x[1])
        simple_anneal = SimulatedAnnealing(x_length=2, x_bounds=(-10, 10), objective_function=simple_objective)
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

