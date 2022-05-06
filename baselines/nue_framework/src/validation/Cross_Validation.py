from abc import ABC, abstractmethod


class Cross_Validation(ABC):   
    
    @abstractmethod
    def split(self, X, y = None):
        pass
    
    @abstractmethod
    def __iter__(self):
        self.i = 0
        return self
    
    @abstractmethod
    def __next__(self):
        pass