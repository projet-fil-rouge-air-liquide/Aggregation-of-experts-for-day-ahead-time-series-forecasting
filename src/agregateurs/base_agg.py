from abc import ABC, abstractmethod

class BaseAgg(ABC):
    def __init__(self,experts,name):
        self.experts = experts
        self.name = name
        self.agregateur = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_agg, y):
        pass
    
    @abstractmethod
    def predict(self,X_agg):
        pass 
