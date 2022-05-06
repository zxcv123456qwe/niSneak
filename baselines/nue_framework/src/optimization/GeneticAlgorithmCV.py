from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from optimization import grid_to_bounds, grid_types, cast_parameters, aggregate_dict, random_population, UnknownParameterTypeError
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from optimization import BaseOptimizer

# GeneticAlgorithm
# Based on the implementation by Ryan (Mohammad) Solgi
# https://pypi.org/project/geneticalgorithm/
# Adapted to the scikit learn BaseSearchCV class
class GeneticAlgorithmCV(BaseOptimizer):
    def __init__(self, estimator, search_space,
                 max_num_iteration, population_size, mutation_probability,
                 elit_ratio, crossover_probability, parents_portion,
                 crossover_type, max_iteration_without_improv,
                 *, scoring=None, n_jobs=None, refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.search_space = search_space
        
        self.max_num_iteration = max_num_iteration
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.elit_ratio = elit_ratio
        self.crossover_probability = crossover_probability
        self.parents_portion = parents_portion
        self.crossover_type = crossover_type
        self.max_iteration_without_improv = max_iteration_without_improv
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        self.evaluation_function_ = evaluate_candidates
        ga_function = lambda X: self.get_evaluation(X)
        
        bounds = grid_to_bounds(self.search_space)
        categories = dict( [ (k,v) for k,v in self.search_space.items() if k not in bounds.keys() ] )
        types = grid_types(self.search_space)
        
        # Cast to genetic algorithm format
        ga_bounds = list( [types[key](min_v), types[key](max_v)] for key, (min_v, max_v) in bounds.items() )
        
        ga_types = [ ["real"] if "float" in types[key].__name__ else ["int"] for key in bounds.keys() ]
        
        # For categories, convert into int
        # This may add the false impression of locality
        for param, values in categories.items():
            ga_bounds.append( [0, len(values) - 1] )
            ga_types.append( ["int"] )
        
        # Save names of features to build dictionaries
        self.bounds_ = bounds
        self.categories_ = categories
        self.types_ = types
        
        ga_length = len(ga_bounds)
        ga_bounds = np.array(ga_bounds)
        ga_types = np.array(ga_types)
        
        # Parameters
        ga_params = {
            "max_num_iteration" : self.max_num_iteration,
            "population_size" : self.population_size,
            "mutation_probability" : self.mutation_probability,
            "elit_ratio" : self.elit_ratio,
            "crossover_probability" : self.crossover_probability,
            "parents_portion" : self.parents_portion,
            "crossover_type" : self.crossover_type,
            "max_iteration_without_improv" : self.max_iteration_without_improv
            }
        
        # Instantiate and run GA
        model = ga(function=ga_function, dimension=ga_length,
                variable_type_mixed=ga_types, variable_boundaries=ga_bounds,
                algorithm_parameters=ga_params)
        model.run()
        
    
    # Interface method for genetic algorithm
    # Evaluates the provided metric, and returns its mean score
    # Maintains a cache to not re-evaluate parameters
    def get_evaluation(self, params):
        # First, we need to convert to dictionary
        
        # Count how many of each type of feature we have
        tot_par = len(params)
        num_par = len( self.bounds_ )
        cat_par = tot_par - num_par
        # Sanity check
        if len(self.categories_) != cat_par:
            # A parameter was neither numerical nor categorical
            raise UnknownParameterTypeError
        
        # True numerical features come first in the array
        # Convert those to their literal values
        params_dict = dict( zip( self.bounds_.keys(), params[ 0 : num_par ] ) )
        for key, val in params_dict.items():
            # Convert to correct type
            params_dict[key] = self.types_[key](val)
        
        # For numerical features, we convert to their actual value
        for i, d in zip( range(num_par, tot_par), range(0, cat_par) ):
            par_idx = int(params[i])
            par_name = list(self.categories_.keys())[d]
            params_dict[par_name] = self.categories_[par_name][par_idx]
        
        # Evaluate the new parameter configuration
        # The evaluation_function_ attribute should be defined before calling
        self.eval_cache_ = self.evaluation_function_( [params_dict] )
        
        # Search for the performance of the new metric
        idx = self.eval_cache_["params"].index( params_dict )
        return self.eval_cache_[self.mean_test_name_][idx] * self.scoring_sign_ * -1 # Loss function