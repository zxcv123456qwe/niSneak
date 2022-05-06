import numpy as np
import os
import sys

class Loader:
    
    def __init__(self, prefix = "config/"):
        self.prefix = prefix
    
    def load_file(self, file, accept_empty = False):
        file = self.prefix + file
        config = []
        treat = []
        param = False #Currently reading parameters
        
        if os.path.isfile(file):
            with open(file, "r") as f:
                for l in f:
                    l = l.strip()
                    if len(l) > 0 and l[0] != "#":
                        l = l.replace("\n", "")
                        if l != "":
                            if l[0] == "-":
                                if param == False:
                                    # Error, void config
                                    config = []
                                    break
                                l2 = l[1:].split(":")
                                if len(l2) != 2:
                                    # Error, void config
                                    config = []
                                    break
                                param, val = l2
                                param = param.strip()
                                param = param.split(",")
                                typ = str
                                if len(param) > 1:
                                    if param[1] == "int":
                                        typ = int
                                    elif param[1] == "float":
                                        typ = float
                                    elif param[1] == "str":
                                        typ = str
                                    elif param[1] == "bool":
                                        typ = bool
                                    else:
                                        # Error, void config
                                        config = []
                                        break
                                
                                val = val.strip()
                                
                                # See if multiple ranges
                                parts = val.split(";")
                                parts2 = []
                                for val in parts:
                                    # See if range of values
                                    val = val.strip()
                                    if len(val) >= 2 and val[0] == "[" and val[-1] == "]":
                                        if typ == int or typ == float:
                                            val = val[1:-1].split(",")
                                            ini = 0
                                            inc = 1
                                            fun = "x"
                                            if len(val) == 1:
                                                end = typ( val[0] )
                                            if len(val) >= 2:
                                                ini = typ( val[0] )
                                                end = typ( val[1] )
                                            if len(val) >= 3:
                                                inc = typ( val[2] )
                                            if len(val) >= 4:
                                                fun = val[3]
                                            if len(val) >= 5:
                                                # Error, void config
                                                config = []
                                                break
                                            eps = 1 if type(inc) == int else sys.float_info.epsilon
                                            val = list((lambda x : eval(fun))(np.arange(ini, end + eps, inc)))
                                        else:
                                            # Error, void config
                                            config = []
                                            break
                                    # If not, sequence of values
                                    else:
                                        val = val.split(",")
                                        for i in range(len(val)):
                                            val[i] = val[i].strip()
                                            if val[i] != "None":
                                                try:
                                                    val[i] = typ(val[i])
                                                except:
                                                    pass # Allow strings as parameters
                                            else: val[i] = None
                                    # Save into list
                                    parts2.extend(val)
                                
                                val = parts2
                                if len(val) == 1:
                                    val = val[0]
                                treat[1][param[0]] = val
                            else:
                                if len(treat) > 0:
                                    config.append(treat)
                                    treat = []
                                if l[-1] == ":":
                                    l = l[:-1]
                                    param = True
                                treat.append(l.lower())
                                treat.append({})
            if len(treat) > 0:
                config.append(treat)
                treat = []
        
        if accept_empty and len(config) == 0:
            config.append(["none", {}])
        else:
            if len(config) == 0:
                raise Exception(f"File {file} not found or is empty.")
        
        return config
    
    def load_file_large(self, file):
        contents = self.load_file(file)
        new_contents = {}
        for lis in contents:
            key, val = lis
            pair = key.split(":")
            # Keep normal configs as-is
            if len(pair) == 1:
                new_contents[key] = val
                continue
            if len(pair) != 2:
                # Error, void config
                new_contents = {}
                break
            # strip
            for i in range(len(pair)):
                pair[i] = pair[i].strip()
            cat, tech = pair
            new_contents[ cat ] = (tech, val)
        
        return new_contents
    
    def load_file_simple(self, file, accept_none = True):
        file = self.prefix + file
        config = []
        
        with open(file, "r") as f:
            for l in f:
                l = l.strip()
                if len(l) > 0 and l[0] != "#":
                    l = l.replace("\n", "")
                    if l != "":
                        config.append(l.lower())
        if not(accept_none) and len(config) == 0:
            config.append("None")
        return config
        
    
    def load_config(self):
        FW = self.load_file_large("FW")
        DS = self.load_file("DS")
        PP = self.load_file("PP", True)
        DT = self.load_file("DT", True)
        AS = self.load_file("AS", True)
        PT = self.load_file("PT")
        LA = self.load_file("LA")
        EM = self.load_file("EM")
        
        return FW, DS, PP, DT, AS, PT, LA, EM
    
    def load_config_model(self):
        FW = self.load_file_large("FW")
        MO = self.load_file("MO")
        PT = self.load_file("PT")
        
        return FW, MO, PT
        
        