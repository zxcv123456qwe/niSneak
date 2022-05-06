# Libraries
import os
import pandas as pd

# TODO fix imports
from map import dataset_db
from map import validation_db as cv_db
from reading import Loader
from sklearn.model_selection import KFold

# Prelude
idx_dir = "idx/"
config_dir = "config/fairness/"
groups = 1 # Partition the indexes into different datasets? For multiple PCs

# Load configuration
FW, DS, PP, DT, AS, PT, LA, EM = Loader(config_dir).load_config()
datasets = dataset_db.get(DS)

# Cross Validation
# Defaults to 80:20 train test split
cv = cv_db[ FW["cv"][0] if "cv" in FW.keys() else "traintestsplit" ]( **(FW["cv"][1] if "cv" in FW.keys() else {"n_splits":1, "test_size":0.2}) )


# Create idx directory if doesnt exist
if not os.path.exists( config_dir + "idx/" ):
    os.makedirs(config_dir + "idx/" )


# Dataset Loop
for ds in datasets:
    ds.set_datapath("data/")
    dataframe = ds.get_dataframe()

    # If DS dir doesnt exist in idx, create it
    ds_dir = ds.id.split("/")[0]
    if not os.path.exists( config_dir + "idx/" + ds_dir  + "/" ):
        os.makedirs(config_dir + "idx/" + ds_dir + "/" )
    
    X = dataframe[ dataframe.columns.difference([ds.predict]) ]
    Y = dataframe[ ds.predict ]
    
    train_set = []
    test_set = []
    
    # Cross validation
    for iteration, (train_index, test_index) in enumerate(cv.split(dataframe)):
        
        # Save indices
        train_set.append(list(train_index))
        test_set.append(list(test_index))
    
    
    
    df_train = pd.DataFrame( train_set )
    df_test = pd.DataFrame( test_set )
    
    if groups == 1:
        # Save the indexes
        # Use dataset name
        df_train.to_csv( config_dir + idx_dir + ds.id + "_train.csv", index=False, header = False )
        df_test.to_csv( config_dir + idx_dir + ds.id + "_test.csv", index=False, header = False )
    elif groups > 1:
        kf = KFold(groups, shuffle=False)
        for i, (_, idx) in enumerate(kf.split(df_train, df_test)):
            sub_train = df_train.iloc[idx,:]
            sub_test = df_test.iloc[idx,:]
            # Save the indexes
            # Use dataset name
            sub_train.to_csv( config_dir + idx_dir + ds.id + "_train_"+str(i)+".csv", index=False, header = False )
            sub_test.to_csv( config_dir + idx_dir + ds.id + "_test_"+str(i)+".csv", index=False, header = False )
            
            
    