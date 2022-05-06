import numpy as np
import numbers
import sys

class UnknownParameterTypeError(Exception):
    pass

def grid_to_bounds(param_grid):
    new_params = {}
    for key, params in param_grid.items():
        params_float = np.array(params)[[ isinstance(x, numbers.Number) for x in params ]].astype(np.float)
        if params_float.size > 0:
            new_params[key] = [ np.min(params_float), np.max(params_float) ]
        #else:
        #    new_params[key] = params
    return new_params

def grid_to_bounds_str(param_grid):
    new_params = {}
    for key, params in param_grid.items():
        params_float = np.array(params)[[ isinstance(x, numbers.Number) for x in params ]].astype(np.float)
        if params_float.size > 0:
            new_params[key] = [ np.min(params_float), np.max(params_float) ]
        else: # strings
           new_params[key] = [ 0, len(params) - sys.float_info.epsilon ]
    return new_params

def grid_types(param_grid):
    types = {}
    for key, params in param_grid.items():
        types[key] = type(params[0])
    return types

def types_as_str(types):
    new_types = {}
    for t, v in types.items():
        nv = "s" # string
        if v in [float, np.float32, np.float64]:
            nv = "f"
        if v in [int, np.int32, np.int64]:
            nv = "i"
        new_types[t] = nv
    return new_types

def cast_parameters(params, types):
    if len(params) == len(types):
        result = []
        for i in range(len(params)):
            result.append(types[i](params[i]))
        return result
    return params

def aggregate_dict(list_of_dict):
    res = {}
    if len(list_of_dict) > 0:
        keys = list(list_of_dict[0].keys())
        for key in keys:
            elem = []
            for dictt in list_of_dict:
                elem.append( dictt[key] )
            res[key] = elem
    return res

def unaggregate_dict(dict_of_lists, idx = None):
    if idx == None:
        idx = [ i for i in range( len(dict_of_lists[ dict_of_lists.keys()[0] ]) ) ]
    lis = []
    for i in idx:
        d = {}
        for k in dict_of_lists.keys():
            d[k] = dict_of_lists[k][i]
        lis.append(d)
    return lis

def random_population(numerical, numerical_types, categorical, size):
    dimensions = len(numerical)
    
    # Normalized numerical values
    population = [dict(zip(numerical.keys(), np.random.rand(dimensions))) for i in range(size)]
    
    # Denormalize
    min_b, max_b = np.asarray(list(numerical.values()))[:,0], np.asarray(list(numerical.values()))[:,1]
    diff = np.fabs(min_b - max_b)
    population = np.array([dict(zip(ind.keys(), cast_parameters(min_b + np.array(list(ind.values())) * diff, numerical_types) )) for ind in population])
    
    # Add categorical features
    population = np.array([ dict( zip( list(ind.keys()) + ( list(categorical.keys()) ),
                                      list(ind.values()) + ([ np.random.choice(list(val)) for val in categorical.values() ]) ) )
                           for ind in population ])
    
    return population

def zip_one( pars, vals ):
    return dict(zip(pars, vals))

def zip_many( pars, indivs ):
    return [ zip_one(pars, ind) for ind in indivs ]

