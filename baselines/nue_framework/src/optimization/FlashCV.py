from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from optimization import grid_to_bounds, grid_types, cast_parameters, aggregate_dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from optimization import BaseOptimizer

# FLASH
# Based on the implementation by Tianpei Xia
# OIL framework
# https://github.com/arennax/effort_oil_2019
# Adapted to the scikit learn BaseSearchCV class
class FlashCV(BaseOptimizer):
    def __init__(self, estimator, search_space, budget, population_size, initial_size,
                 *, scoring=None, n_jobs=None, refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.search_space = search_space
        self.budget = budget
        self.population_size = population_size
        self.initial_size = initial_size
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        bounds = grid_to_bounds(self.search_space)
        types = grid_types(self.search_space)
        types = dict( [ (key, val) for key, val in types.items() if key in bounds.keys() ] ) # Only for bounded types
        types = list(types.values())
        dimensions = len(bounds)
        
        # Our initial population is completely random (normalized)
        population = [dict(zip(bounds.keys(), np.random.rand(dimensions))) for i in range(self.population_size)]
        
        # Scale the population to the hyper-parameter values
        min_b, max_b = np.asarray(list(bounds.values()))[:,0], np.asarray(list(bounds.values()))[:,1]
        diff = np.fabs(min_b - max_b)
        population = np.array([dict(zip(ind.keys(), cast_parameters(min_b + np.array(list(ind.values())) * diff, types) )) for ind in population])
        
        # Sample to get a modeling pool and a candidate pool
        samples = np.random.randint( self.population_size, size=self.initial_size )
        modeling_pool = population[samples]
        candidate_pool = np.delete(population, samples)
        
        # Get evaluation values
        results = evaluate_candidates(modeling_pool)
        
        fitness = np.array(results[self.mean_test_name_])
        # Create dataframe
        data = pd.DataFrame.from_dict( aggregate_dict(modeling_pool) )
        
        model = DecisionTreeRegressor()
        for iteration in range(self.initial_size, self.budget):
            # Fit model and try to obtain "best" estimator from candidates
            model.fit(data, fitness)
            candidate_fitness = model.predict( pd.DataFrame.from_dict(aggregate_dict(candidate_pool)) )
            candidate_fitness = candidate_fitness * self.scoring_sign_
            
            # Candidate has best estimated fitness
            next_idx = np.argmax( candidate_fitness )
            next_element = candidate_pool[next_idx]
            
            # Remove candidate from pool and add to modeling pool
            candidate_pool = np.delete(candidate_pool, next_idx)
            modeling_pool = np.append( modeling_pool, [next_element] )
            
            # Evaluate new candidate and obtain its fitness
            results = evaluate_candidates([next_element])
            fitness = np.array(results[self.mean_test_name_])
            data = pd.DataFrame.from_dict( aggregate_dict(modeling_pool) ) # Re-make training data
            
            
