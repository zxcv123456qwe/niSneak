from map import Database
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile,\
    SelectFpr, SelectFdr, SelectFwe, RFE, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from selection import AttributeSelector, VariantThreshold, BorutaSelector,\
    DummySelector

# =============================================================================
# 
# selection_functions = [chi2, f_classif, f_regression]
# 
# as_kbestparams = {"score_func" : selection_functions, "k" : [5,10,15,20]}
# as_kbest = AttributeSelector("kbest", SelectKBest, as_kbestparams)
# 
# as_percentileparams = {"score_func" : selection_functions, "percentile" : [5,10,15,20]}
# as_percentile = AttributeSelector("percentile", SelectPercentile, as_percentileparams)
# 
# as_fprparams = {"score_func" : selection_functions}
# as_fpr = AttributeSelector("fpr", SelectFpr, as_fprparams)
# 
# as_fdrparams = {"score_func" : selection_functions}
# as_fdr = AttributeSelector("fdr", SelectFdr, as_fdrparams)
# 
# as_fweparams = {"score_func" : selection_functions}
# as_fwe = AttributeSelector("fwe", SelectFwe, as_fweparams)
# =============================================================================

selection_db = Database(AttributeSelector, {"none":DummySelector,
                                     "kbest":SelectKBest,
                                     "percentile":SelectPercentile,
                                     "fpr":SelectFpr,
                                     "fdr":SelectFdr,
                                     "fwe":SelectFwe,
                                     "rfe":RFE,
                                     #"variancethreshold":VarianceThreshold,
                                     "variancethreshold":VariantThreshold,
                                     "correlationkbest":SelectKBest,
                                     "correlationpercentile":SelectPercentile,
                                     "randomforest":BorutaSelector,
                                     "boruta":BorutaSelector},
                 {"rfe": {"wrapper":True},
                  "boruta":{"wrapper":True},
                  "correlationkbest":{"score_func":f_regression},
                  "correlationpercentile":{"score_func":f_regression},
                 "randomforest":{"wrapper":False, "estimator":RandomForestRegressor(n_jobs=-1, max_depth=5), "n_estimators":"auto"}
                 })


