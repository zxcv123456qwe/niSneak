from sklearn.model_selection._search import BaseSearchCV
import numpy as np
from abc import ABCMeta
from optimization import grid_to_bounds, grid_types, cast_parameters, aggregate_dict, random_population
import pandas as pd
from sklearn.model_selection import train_test_split
from optimization import HarmonySearch
from optimization import BaseOptimizer

class HarmonySearchCV(BaseOptimizer):
    
    def __init__(self, estimator, search_space, memory_size, memory_considering_rate,
                 pitch_adjustment_rate, fret_width, max_steps, *, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True, **kwargs):
        
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        
        self.search_space = search_space
        
        # Harmony search parameters
        self.memory_size = memory_size
        self.memory_considering_rate = memory_considering_rate
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.fret_width = fret_width
        self.max_steps = max_steps
    
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"
        self.scoring_sign_ = self.scoring[self.refit]._sign if self.multimetric_ else self.scoring._sign
        
        # Instantiate searcher object
        self.hs_ = self.HarmonySearcher( hms = self.memory_size,
            hmcr = self.memory_considering_rate, par = self.pitch_adjustment_rate,
            fw = self.fret_width, max_steps = self.max_steps,
            search_space = self.search_space, target_function = evaluate_candidates,
            sign = self.scoring_sign_, max_score = None, scoring_name = self.mean_test_name_)
        
        # Run harmony search
        self.hs_.run(verbose=False)
    
    
    class HarmonySearcher(HarmonySearch):
        
        def __init__(self, hms, hmcr, par, fw, max_steps,
                     search_space, target_function, sign, max_score=None, scoring_name = "mean_test_score"):
            
            super().__init__(hms, hmcr, par, fw, max_steps, max_score)
            
            self.search_space = search_space
            self.target_function = target_function
            self.sign = sign
            self.scoring_name = scoring_name
            
            # A cache of scores to not run evaluate_candidates with each new score
            self.harmonies_, self.scores_ = [], []
            
            # Search space is saved separately for categories and numbers
            self.bounds = grid_to_bounds(self.search_space)
            self.categories = dict( [ (k,v) for k,v in self.search_space.items() if k not in self.bounds.keys() ] )
            self.types = grid_types(self.search_space)
        
        def _random_harmony(self):
            # In our case, we generate harmonies for each numerical parameter
            # Harmonies are normalized in [0, 1[ range
            # When used to create a model, they are cast to the appropriate range
            
            return list(np.random.random( len(self.bounds) ))
            
        
        def _score(self, harmony):
            score = None
            
            # First, check if we have visited these parameters
            if harmony in self.harmonies_:
                
                # Return the score we previously reported
                idx = self.harmonies_.index( harmony )
                score = self.scores_[idx]
            
            else:
                # Construct parameter vector from harmony
                parameters = {}
                
                for i, (hp, vals) in enumerate(self.bounds.items()):
                    min_bound = vals[0]
                    diff = vals[1] - vals[0]
                    
                    # If the type is int, we need to add 1, as we round down
                    if self.types[ hp ] in [np.int64, np.int32, int]:
                        diff += 1
                    
                    new_val = min_bound + harmony[i] * diff
                    new_val = self.types[ hp ]( new_val )
                    
                    # Add to our parameters
                    parameters[ hp ] = new_val
                
                # Evaluate the parameter and store
                results = self.target_function([parameters])
                
                # The score is the last from mean test score
                # It uses the max, so we invert if using a minimizing function
                score = results[self.scoring_name][-1] * self.sign
                
                # The new parameter and scores are added
                self.harmonies_.append(harmony)
                self.scores_.append(score)
            
            return score
            
        
        
    
