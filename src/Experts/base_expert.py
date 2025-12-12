from abc import ABC, abstractmethod

class BaseExpert(ABC):
    def __init__(self,features,name):
        self.features = features
        self.name = name
        self.expert = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self,X,y):
       pass
    @abstractmethod
    def predict(self,X):
        pass


