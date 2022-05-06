import numpy as np
from utils import zitler_dominates, normalize_score

def get_pareto_front(cv_results, metrics):
    cv_names = [ "mean_test_" + metric.name for metric in metrics ]
    cv_metrics = [ cv_results[ name ] for name in cv_names ]
    pareto_front = []
    
    # Now, search for the pareto front
    # 1) No solution in the front is strictly better than any other
    # 2) Solutions that are strictly worse are removed
    
    # We do this process for each explored hyper-parameter
    for i in range(len( cv_metrics[0] )):
        included = True # We start assuming the current parameter can be included
        
        # Now, check for each of the pareto-front
        # Whether it is overshadowed by any other parameter
        for fp in pareto_front:
            overshadowed = True # Assume it is, until we find a case it isnt
            
            # Check for each metric
            for m_object, metric in zip(metrics, cv_metrics):
                # Gets around Nan values
                if True in np.isnan(metric): break
                
                sign = m_object._sign # Metric sign
                
                # Check if metric is overshadowed
                overshadowed = overshadowed and ( metric[fp] * sign >= metric[i] * sign )
                if not overshadowed: break # If we found it is not, stop searching
            
            # If the parameter is overshadowed by someone in the front, it is not included
            included = included and not(overshadowed)
            if not included: break # End if we already found it its not included
        
        # If the metric was not overshadowed by anyone, its a new point
        # Now, find out if the new point overshadows some of the existing points
        if included:            
            for fp in pareto_front:
                overshadowed = True # Assume it is, until we find a case it isnt
                
                # Check for each metric
                for m_object, metric in zip(metrics, cv_metrics):
                    sign = m_object._sign # Metric sign
                    
                    # Check if metric is overshadowed
                    overshadowed = overshadowed and ( metric[i] * sign >= metric[fp] * sign )
                    if not overshadowed: break # If we found it is not, stop searching
                
                # If it is overshadowed by the new point, remove
                if overshadowed: pareto_front.remove(fp)
            
            # Lastly, add the new point to the front
            pareto_front.append(i)
    
    return pareto_front


def get_pareto_front_zitler(cv_results, metrics):
    cv_names = [ "mean_test_" + metric.name for metric in metrics ]
    cv_metrics = [ cv_results[ name ] for name in cv_names ]
    pareto_front = []

    cv_metrics = [ [ cv_metrics[j][i] for j in range(len(cv_metrics)) ] for i in range(len(cv_metrics[0])) ]
    norm_metrics = [ normalize_score(cvr, metrics) for cvr in cv_metrics ]
    
    # Now, search for the pareto front
    # 1) No solution in the front is strictly better than any other
    # 2) Solutions that are strictly worse are removed
    
    # We do this process for each explored hyper-parameter
    for i in range(len( norm_metrics )):
        included = True # We start assuming the current parameter can be included
        current = norm_metrics[i]

        # Now, check for each of the pareto-front
        # Whether it is overshadowed by any other parameter
        for fp in pareto_front:
            
            # Check if this member of pareto front has better zitler than new point
            fpp = norm_metrics[fp]
            overshadowed = zitler_dominates( fpp, current ) == 1
            
            # If the parameter is overshadowed by someone in the front, it is not included
            included = included and not(overshadowed)
            if not included: break # End if we already found it its not included
        
        # If the metric was not overshadowed by anyone, its a new point
        # Now, find out if the new point overshadows some of the existing points
        if included:            
            for fp in pareto_front:
                overshadowed = True # Assume it is, until we find a case it isnt

                fpp = norm_metrics[fp]
                overshadowed = zitler_dominates( current, fpp ) == 1
                
                # If it is overshadowed by the new point, remove
                if overshadowed: pareto_front.remove(fp)
            
            # Lastly, add the new point to the front
            pareto_front.append(i)
    
    return pareto_front

