import pandas as pd
from utils import ps

class Dataset(ps):
    data_path = ""
    
    def __init__(self, name, dummy, params):
        self.name = name
        self.formatt = name.split(".")[-1] if name.find(".") != 1 else None
        self.id = ".".join(name.split(".")[0:-1]) if name.find(".") != 1 else name
        self.params = params
        
        self.predict = params["predict"].lower()

        problem = "None"
        if "problem" in params.keys():
            problem = params["problem"].lower()
        self.problem = problem
        
        secondary = []
        if "secondary" in params.keys():
            secondary = params["secondary"]
        if type(secondary) != list:
            secondary = [secondary]
        self.secondary = list(secondary)

        exclude = []
        if "exclude" in params.keys():
            exclude = params["exclude"]
        self.exclude = list(exclude)
        
        drop = []
        if "drop" in params.keys():
            drop = params["drop"]
        self.drop = [int(i) for i in list(drop)]
        
        delimiter = ","
        if "delimiter" in params.keys():
            delimiter = params["delimiter"]
        self.delimiter = delimiter
        
        na_values = "NA"
        if "na_values" in params.keys():
            na_values = params["na_values"]
        self.na_values = na_values
        
        as_dummies = False
        if "as_dummies" in params.keys():
            as_dummies = params["as_dummies"]
        self.as_dummies = as_dummies
        
        quotechar = '"'
        if "quotechar" in params.keys():
            quotechar = params["quotechar"]
        self.quotechar = quotechar

    def get_dataframe(self):
        if self.formatt.lower() == "csv":
            df = pd.read_csv(self.data_path + self.name, delimiter = self.delimiter, na_values = self.na_values, quotechar = self.quotechar)
        new_c = []
        for cn in df.columns: new_c.append(cn.lower())
        df.columns = new_c
        if df is not None:
            return self.preprocess(df)
    
    def preprocess(self, df):
        if self.exclude != []:
            df = df[ df.columns.difference(self.exclude) ]
        if self.drop != []:
            df = df.drop(index=self.drop)
        #if self.as_dummies:
        #    df = pd.get_dummies(df, dtype=np.bool)
        return df
    
    def set_datapath(self, path):
        self.data_path = path
    
    def get_train(self):
        res = Dataset(self.id + "_train." + self.formatt, None, self.params)
        res.set_datapath(self.data_path)
        return res
    
    def get_test(self):
        res = Dataset(self.id + "_test." + self.formatt, None, self.params)
        res.set_datapath(self.data_path)
        return res