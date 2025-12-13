from lightgbm import LGBMRegressor,early_stopping
from src.experts.base_expert import BaseExpert
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os


class LGBMExpert(BaseExpert):
    def __init__(
        self,
        features=None,
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
        ):
        # initialisation de la classe mère        if features is None:
            features = [
                "speed_longitudinale_100m", 
                "speed_latitudinale_100m",
                "2m_temperature", 
                "mean_sea_level_pressure", 
                "sea_surface_temperature",
                "surface_pressure", 
                "Wind_Norm", 
                "Wind_Norm_Cubes", 
                "wind_std_3h",
                "wind_cv_3h", 
                "Wind_Norm_lag_1h", 
                "Wind_Norm_lag_24h",
                "Hour_sin",
                "Hour_cos",
                "Weekday_sin",
                "Weekday_cos",
                "Month_sin",
                "Month_cos",
                "P_curve",
                "Wind_mean_3h",
                "Air_density",
                "Wind_Dir_Meteo_sin",
                "Wind_Dir_Meteo_cos"
            ]
        # paramètres spécifiques de la classe LGBM
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample=subsample
        self.colsample_bytree = colsample_bytree
        self.random_state=random_state
        # le modèle LGBM
        self.expert = LGBMRegressor(
            n_estimators = self.n_estimators,
            learning_rate = self.learning_rate,
            num_leaves = self.num_leaves, 
            max_depth = self.max_depth, 
            min_child_samples = self.min_child_samples,
            subsample = self.subsample,
            colsample_bytree = self.colsample_bytree,
            random_state = self.random_state
        )

    def fit(self,X,y):
        X_sel = X[self.features]
        self.expert.fit(X_sel,y)
        self.is_fitted = True

    def predict(self,X):
        X_sel = X[self.features]
        return self.expert.predict(X_sel)
    
    pass

