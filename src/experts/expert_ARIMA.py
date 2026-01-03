# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAExpert():
    def __init__(
        self,
        features=None,
        order=(2, 0, 2),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    ):
        if features is None:
            features = [
                "speed_longitudinale_100m",
                "speed_latitudinale_100m",
                "2m_temperature",
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
                "Wind_mean_3h",
                "Air_density",
                "Wind_Dir_Meteo_sin",
                "Wind_Dir_Meteo_cos"
            ]
        self.features = features
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

        self.model = None
        self.results = None
        self.is_fitted = False

    def fit(self, X, y):
        """
        X : DataFrame des features exogènes
        y : Series target
        """
        X_sel = X[self.features]

        self.model = SARIMAX(
            endog=y,
            exog=X_sel,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )

        self.results = self.model.fit(disp=False)
        self.is_fitted = True

        return self

    def predict(self, X):
        """
        X : DataFrame des features exogènes futures
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle ARIMA n'est pas entraîné.")

        X_sel = X[self.features]

        forecast = self.results.predict(
            start=len(self.results.fittedvalues),
            end=len(self.results.fittedvalues) + len(X_sel) - 1,
            exog=X_sel
        )

        return np.asarray(forecast)

    @property
    def aic_(self):
        return self.results.aic

    @property
    def bic_(self):
        return self.results.bic

    def summary(self):
        return self.results.summary()
