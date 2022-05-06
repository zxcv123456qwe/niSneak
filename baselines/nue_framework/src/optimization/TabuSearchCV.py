from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from abc import ABCMeta
from optimization import grid_to_bounds, grid_types, cast_parameters, aggregate_dict, random_population
import pandas as pd
from sklearn.model_selection import train_test_split
from optimization import TabuSearch
from optimization import BaseOptimizer

# Tabu Search for a NeverGrad algorithm
# Using the implementation of the Solid library
# https://github.com/100/Solid
# Adapted to the scikit learn BaseSearchCV class

class TabuSearchCV(BaseOptimizer):
    
    def __init__(self, estimator, search_space, tabu_size, max_steps,
                 neighborhood_size = 5, *, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True, **kwargs):
        
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        
        self.search_space = search_space
        self.tabu_size = tabu_size
        self.max_steps = max_steps
        self.neighborhood_size = neighborhood_size
        
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        # Our states are dictionaries of parameters
        # Initial state is default parameters
        default_params = dict( zip( self.search_space.keys(),
           [ self.estimator.get_params()[key] for key in self.search_space.keys() ] ) )
        
        # Instantiate searcher object
        self.ts_ = self.TabuSearcher( self.search_space, default_params=default_params,
                                tabu_size=self.tabu_size, max_steps=self.max_steps,
                                neighborhood_size=self.neighborhood_size,
                                target_function=evaluate_candidates, sign=self.scoring_sign_,
                                scoring_name = self.mean_test_name_)
        
        # Run tabu search
        self.ts_.run(verbose=False)


    # Sub-class searcher
    # This one inherits from Solid
    class TabuSearcher(TabuSearch):
        def __init__(self, search_space, default_params, tabu_size,
                     max_steps, neighborhood_size, target_function, sign, scoring_name):
            
            super().__init__(initial_state = default_params,
                                tabu_size = tabu_size, max_steps = max_steps,
                                max_score=None)
            
            self.search_space = search_space
            self.neighborhood_size = neighborhood_size
            self.target_function = target_function
            self.sign = sign
            self.scoring_name = scoring_name
            
            # A cache of scores to not run evaluate_candidates with each new score
            self.parameters_, self.scores_ = [], []
        
        def _score(self, state):
            score = None
            
            # First, check if we have visited these parameters
            if state in self.parameters_:
                
                # Return the score we previously reported
                idx = self.parameters_.index( state )
                score = self.scores_[idx]
            
            else:
                # Evaluate the parameter and store
                results = self.target_function([state])
                
                # The score is the last from mean test score
                # It uses the max, so we invert if using a minimizing function
                score = results[self.scoring_name][-1] * self.sign
                
                # The new parameter and scores are added
                self.parameters_.append(state)
                self.scores_.append(score)
            
            return score
        
        def _neighborhood(self):
            # Referential to another function
            # In the future, a parameter could be type of neighborhood function
            
            return self._corazza_svr_neighborhood()
        
        # Randomly determine new neighbors
        def _random_neighborhood(self):
            
            
            # Determine seach bounds
            bounds = grid_to_bounds(self.search_space)
            categories = dict( [ (k,v) for k,v in self.search_space.items() if k not in bounds.keys() ] )
            types = grid_types(self.search_space)
            
            # Generate X random new individuals
            population = random_population( bounds, list(types.values()), categories,
                                           self.neighborhood_size )
            
            return population
        
        # Method proposed by Corazza et al, 2013
        # Corazza, A., Di Martino, S., Ferrucci, F., Gravino, C., Sarro, F., & Mendes, E. (2013).
        # Using tabu search to configure support vector regression for effort estimation.
        # Empirical Software Engineering, 18(3), 506-546.
        # 
        # For each parameter, 80% chance of doing +- 20% (categoricals are kept)
        # The remaining 20% is to randomly determine the value
        # Search space will be respected
        #
        # Generates X neighbors
        # Configured by class attribute
        def _corazza_svr_neighborhood(self):
            
            # Sanity check
            # If best parameters are not set, use random population
            if not hasattr(self, "best"):
                return self._random_neighborhood()
            
            # Get bounds and categories list
            bounds = grid_to_bounds(self.search_space)
            categories = dict( [ (k,v) for k,v in self.search_space.items() if k not in bounds.keys() ] )
            types = grid_types(self.search_space)
            
            neighbors = []
            
            for i in range(self.neighborhood_size):
                
                new_ind = {}
                
                # Generate numerical attributes
                for hp, vals in bounds.items():
                    # 80% chance of keeping, 20% of random parameter
                    # Done by altering bounds
                    
                    if np.random.random() < 0.8: # Keep best, respecting bounds
                        # We alter the parameter by +- 20% of its current value
                        
                        # If it is none or str, keep it
                        current_hp = self.best[hp]
                        if current_hp is None or type(current_hp) is str:
                            new_ind[hp] = current_hp
                            continue
                        
                        min_bound = current_hp * 0.8
                        min_bound = max( vals[0], min_bound ) # Minimum bound
                        max_bound = current_hp * 1.2
                        max_bound = max( vals[1], max_bound ) # Maximum bound
                    else: # Random selection within bounds
                        min_bound = vals[0]
                        max_bound = vals[1]
                    
                    # Generate the random number and cast to type
                    number = min_bound + np.random.random() * ( max_bound - min_bound )
                    number = types[hp](number)
                    new_ind[hp] = number
                
                # Generate categorical attributes
                for hp, vals in categories.items():
                    # 80% chance of keeping, 20% of random parameter
                    if np.random.random() < 0.8: # Keep best
                        new_ind[hp] = self.best[hp]
                    else: # Random choice
                        new_ind[hp] = np.random.choice( vals )
    
                neighbors.append(new_ind)
            
            return neighbors