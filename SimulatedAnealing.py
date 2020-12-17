import numpy as np

class SimulatedAnnealing:
    def __init__(self, x_length, x_bounds, objective_function, pertubation_method="simple",
                 annealing_schedule="simple_exponential_cooling", halt_definition = "max_n_temperatures",
                 maximum_markov_chain_length=10):

        self.x_length = x_length    # integer containing length of array x
        self.x_bounds = x_bounds    # tuple containing bounds to x
        self.objective_function = objective_function        # TODO add function to class
        self.pertubation_method = pertubation_method
        self.annealing_schedule = annealing_schedule
        self.halt_definition = halt_definition
        self.maximum_markov_chain_length = maximum_markov_chain_length
        self.minimum_number_of_acceptances = round(0.6*maximum_markov_chain_length)   # 0.6 is a heuristic from lectures


        self.archive = []   # TODO choose nice data structure
        self.temperature_history = []
        self.Markov_chain_length = 0    # initialise to 0
        self.number_of_acceptances = 0  # initialise to 0

        if annealing_schedule == "simple_exponential_cooling":
            self.alpha = 0.95   # alpha is a constant in this case

        if halt_definition == "max_n_temperatures":
            self.max_n_temperatures = 10
        else:
            assert halt_definition == "by_improvement"
            # TODO write this
            pass


    def run(self):
        # initialise x and temperature
        x_current = self.initialise_x()
        objective_current = self.objective_function(x_current)
        self.initialise_temperature(x_current, objective_current)
        done = False    # initialise, done = True when the optimisation has completed
        while done is False:
            x_new = self.perturb_x(x_current)
            objective_new = self.objective_function(x_new)
            delta_objective = objective_new - objective_current
            if delta_objective > 0 or np.exp(-delta_objective/self.temperature) > np.random.uniform(1):
                # accept change if there is an improvement, or probabilisticly (based on given temperature)
                x_current = x_new
                objective_current = objective_new
                self.number_of_acceptances += 1
                # TODO add code to update archive
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
            D_max_change = (self.x_bounds[1] - self.x_bounds[0]) * 0.05  # 5% of the range?
            u_random_sample = np.random.uniform(low=-1, high=1, size=self.x_length)
            return np.clip(x + u_random_sample*D_max_change, self.x_bounds[0], self.x_bounds[1])

    def temperature_scheduler(self):
        if self.Markov_chain_length > self.maximum_markov_chain_length or \
                self.number_of_acceptances > self.minimum_number_of_acceptances:
            self.Markov_chain_length = 0    # restart counter
            self.number_of_acceptances = 0  # restart counter
            if self.annealing_schedule == "simple_exponential_cooling":
                self.temperature = self.temperature * self.alpha
            if self.halt_definition == "max_n_temperatures":
                if len(self.temperature_history) > self.max_n_temperatures:
                    done = True     # stopping criteria has been met
                else:
                    done = False
                return done


if __name__ == "__main__":
    simple_objective = lambda x: x**2
    simple_anneal = SimulatedAnnealing(x_length=1, x_bounds=(-10, 10), objective_function=simple_objective)
    simple_anneal.run()

