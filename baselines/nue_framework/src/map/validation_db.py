from sklearn.model_selection import *
from validation import LoaderCV, BootstrapCV, VersionDropout

validation_db = { "repeatedkfold" : RepeatedKFold,
         "leaveoneout" : LeaveOneOut,
         "loo" : LeaveOneOut,
         "kfold" : KFold,
         "shufflesplit" : ShuffleSplit,
         "repeatedtraintest" : ShuffleSplit,
         "traintestsplit" : ShuffleSplit,
         "loader" : LoaderCV,
         "bootstrap" : BootstrapCV,
         "versiondropout" : VersionDropout,}
