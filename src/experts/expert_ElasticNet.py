import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

class ElasticNetExpert():
    def __init__(
        self,
        features=None,
        alphas=None,
        l1_ratios=None,
        n_split=5,
        max_iter=10000
    ):
        if features is None:
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
                "Y_lag_1h",
                "Y_lag_24h",
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
        self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 50)
        self.l1_ratios = l1_ratios if l1_ratios is not None else np.linspace(0.1, 1.0, 10)
        self.cv = TimeSeriesSplit(n_splits=n_split)

        self.model = ElasticNetCV(
            alphas=self.alphas,
            l1_ratio=self.l1_ratios,
            cv=self.cv,
            max_iter=max_iter
        )
        self.is_fitted = False

    def fit(self, X, y):
        X_sel = X[self.features]
        self.model.fit(X_sel, y)
        self.is_fitted = True

    def predict(self, X):
        X_sel = X[self.features]
        return self.model.predict(X_sel)

    @property
    def best_alpha_(self):
        return self.model.alpha_

    @property
    def best_l1_ratio_(self):
        return self.model.l1_ratio_

    def get_coefficients(self):
        return self.model.coef_, self.model.intercept_