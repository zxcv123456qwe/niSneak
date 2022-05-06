import numpy as np
from optimization import grid_to_bounds, grid_types, cast_parameters, aggregate_dict, unaggregate_dict, \
    random_population, types_as_str
import pandas as pd
from sklearn.model_selection import train_test_split
from optimization import BaseOptimizer
from utils import distance_pair, distance_from, argsort, sortarg, zitler_dominates, normalize_score
from random import sample

# SWAY (Sampling the WAY) algorithm
# Adapted for hyper-parameter optimization
# From "Sampling" as a Baseline Optimizer for Search-based Software Engineering, Chen et al.
# https://github.com/ginfung/FSSE/blob/master/Algorithms/sway_sampler.py
# Adapted to the scikit learn BaseSearchCV class
class SWAYCV(BaseOptimizer):
    def __init__(self, estimator, search_space, *, n_samples = 10000, min_group_size = 0.01,
                scoring=None, n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True, **kwargs):
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.search_space = search_space
        self.n_samples = n_samples
        self.min_group_size = min_group_size
    
    def _run_search(self, evaluate_candidates):
        self.multimetric_ = isinstance(self.scoring, dict)
        self.rank_test_name_ = "rank_test_" + self.refit if self.multimetric_ else "rank_test_score"
        self.mean_test_name_ = "mean_test_" + self.refit if self.multimetric_ else "mean_test_score"

        bounds = grid_to_bounds(self.search_space)
        categories = dict( [ (k,v) for k,v in self.search_space.items() if k not in bounds.keys() ] )
        types = grid_types(self.search_space)
        types_bounded = dict( [ (key, val) for key, val in types.items() if key in bounds.keys() ] ) # Only for bounded types
        types = types_as_str(types)

        self.samples = aggregate_dict(random_population( bounds, list(types_bounded.values()), categories, self.n_samples ) )
        self.types = types
        idx = [i for i in range(self.n_samples)]

        candidates = self._sway(evaluate_candidates, idx)
        # evaluate_candidates( unaggregate_dict(self.samples, candidates) )
    
    def _sway(self, evaluate_candidates, idx):
        if len(idx) < (self.min_group_size * self.n_samples):
            return idx
        d1, d2 = [], []
        (west, east), (west_items, east_items) = self._split( idx )

        # Evaluate candidates
        west_d, east_d = unaggregate_dict(self.samples, [west, east])
        cache = evaluate_candidates([west_d, east_d])

        w_idx = cache["params"].index( west_d )
        if not self.multimetric_:
            w_fitness =  [cache[ self.mean_test_name_ ][w_idx]]
        else:
            w_fitness = tuple([ cache[ f'mean_test_{m}' ][w_idx] for m in self.scoring.keys() ])

        e_idx = cache["params"].index( east_d )
        if not self.multimetric_:
            e_fitness =  [cache[ self.mean_test_name_ ][e_idx]]
        else:
            e_fitness = tuple([ cache[ f'mean_test_{m}' ][e_idx] for m in self.scoring.keys() ])
        
        met = list(self.scoring.values() if self.multimetric_ else [self.scoring])
        w_fitness = normalize_score(w_fitness, met)
        e_fitness = normalize_score(e_fitness, met)
        better = zitler_dominates(w_fitness, e_fitness)

        if better >= 0:
            d1 = self._sway(evaluate_candidates, west_items)
        if better <= 0:
            d2 = self._sway(evaluate_candidates, east_items)

        return d1 + d2
        
    
    def _split(self, idx):
        n_idx = len(idx)
        r = sample(idx, 1)[0]
        east = idx[np.argmax(distance_from( self.samples, self.types, idx, r ))]
        west = idx[np.argmax(distance_from( self.samples, self.types, idx, east ))]
        c = distance_pair( self.samples, self.types, east, west )

        d = []
        for i in idx:
            a = distance_pair( self.samples, self.types, east, i )
            b = distance_pair( self.samples, self.types, west, i )
            d.append( (a*a + c*c - b*b) / ( 2*c ) )
        sorted_idx = sortarg( idx, argsort(d) )
        m = int(n_idx / 2)
        east_items = sorted_idx[0:m]
        west_items = sorted_idx[m:n_idx]

        return (west, east), (west_items, east_items)
