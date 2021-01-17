from rana import rana_func

Comp_config = {"objective_function": rana_func,
               "x_bounds" : (-500, 500),
                "x_length" : 5,
                "parent_number" : 10,
                "child_to_parent_ratio" : 7,
                "bound_enforcing_method" : "not_clipping" ,
                "selection_method" :  "standard_mew_comma_lambda" ,
                "standard_deviation_clipping_fraction_of_range" : 0.01,
                "mutation_covariance_initialisation_fraction_of_range" : 0.01 ,
                "mutation_method" : "complex",
               "termination_min_abs_difference": 1e-6}

Diag_config = {"objective_function": rana_func,
               "x_bounds" : (-500, 500),
                "x_length" : 5,
                "parent_number" : 10,
                "child_to_parent_ratio" : 7,
                "bound_enforcing_method" : "not_clipping" ,
                "selection_method" :  "standard_mew_comma_lambda" ,
                "standard_deviation_clipping_fraction_of_range" : 0.01,
                "mutation_covariance_initialisation_fraction_of_range" : 0.01 ,
                "mutation_method" : "diagonal",
               "termination_min_abs_difference": 1e-6}

Simple_config = {"objective_function": rana_func,
               "x_bounds" : (-500, 500),
                "x_length" : 5,
                "parent_number" : 10,
                "child_to_parent_ratio" : 7,
                "bound_enforcing_method" : "not_clipping" ,
                "selection_method" :  "standard_mew_comma_lambda" ,
                "standard_deviation_clipping_fraction_of_range" : 0.01,
                "mutation_covariance_initialisation_fraction_of_range" : 0.01 ,
                "mutation_method" : "simple",
               "termination_min_abs_difference": 1e-6}
