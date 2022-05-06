class FeatureSelection:
    
    def __init__(self, technique = None, isWrapper = False, **param):
        self.technique = technique
        self.isWrapper = isWrapper
        self.param = param
