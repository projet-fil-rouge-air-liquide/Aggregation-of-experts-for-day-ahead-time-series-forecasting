import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.agregateurs.base_agg import BaseAgg

class AGG_LIN(BaseAgg):
    def __init__(
        self,
        experts,
    ):
        # initialisation de la classe mère
        super().__init__(experts,name="Linéaire")
        # paramètres de la classe
        self.pipeline = Pipeline([
            ("scaler",StandardScaler()),
            ("model",LinearRegression())
        ])

    def fit(self,X_agg,y):
        self.pipeline.fit(X_agg,y)
        self.is_fitted = True

    def predict(self,X_agg):
        return self.pipeline.predict(X_agg)

