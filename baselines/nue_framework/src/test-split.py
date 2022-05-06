# Libraries
import os
from sys import path

from map import dataset_db
from map import validation_db as cv_db
from reading import Loader
from sklearn.model_selection import KFold

# Prelude
data_dir = "data/"
config_dir = "config/see/"
groups = 1 # Partition the indexes into different datasets? For multiple PCs

# Load configuration
FW, DS, PP, DT, AS, PT, LA, EM = Loader(config_dir).load_config()
datasets = dataset_db.get(DS)

# Cross Validation
# Defaults to 80:20 train test split
#cv = cv_db[ FW["cv"][0] if "cv" in FW.keys() else "traintestsplit" ]( **(FW["cv"][1] if "cv" in FW.keys() else {"n_splits":1, "test_size":0.2}) )
cv = (cv_db["traintestsplit"])(n_splits=1, test_size=0.1)

# Dataset Loop
for ds in datasets:
    ds.set_datapath("data/")
    dataframe = ds.get_dataframe()
    
    # Data pre-processing
    # dataframe = prep.process(dataframe)
    
    # Cross validation
    for train_index, test_index in cv.split(dataframe):
        
        # Save indices
        train_set = dataframe.iloc[train_index,:]
        test_set = dataframe.iloc[test_index,:]

        # Save the indexes
        # Use dataset name
        train_set.to_csv( data_dir + ds.id + "_train.csv", index=False )
        test_set.to_csv( data_dir + ds.id + "_test.csv", index=False )
            
    