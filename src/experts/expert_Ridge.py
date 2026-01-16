import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

class RidgeExpert():
    def __init__(
        self,
        features=None,
        n_split=5,
        alphas=None,
    ):
        if features is None:
            features = [
                "speed_longitudinale_100m", 
                "speed_latitudinale_100m",
                "surface_pressure", 
                "Wind_Norm", 
                "Wind_Norm_Cubes", 
                "wind_std_3h",
                "wind_cv_3h", 
                "Wind_Norm_lag_1h", 
                "Wind_Norm_lag_24h",
                "Hour_sin",
                "Hour_cos",
                "Wind_mean_3h",
                "Wind_Dir_Meteo_sin",
                "Wind_Dir_Meteo_cos"
            ]

        self.features = features
        self.alphas = alphas if alphas is not None else np.logspace(-3, 1, 50)
        self.cv = TimeSeriesSplit(n_splits=n_split)
        # modèle ridge
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(
                alphas=self.alphas,
                scoring="neg_mean_absolute_error",
                cv=self.cv
            ))
        ])
        self.is_fitted=False

    def fit(self, X, y):
        X_sel = X[self.features]
        self.pipeline.fit(X_sel, y)
        self.is_fitted = True

    def predict(self, X):
        X_sel = X[self.features]
        return self.pipeline.predict(X_sel)

    @property
    def best_alpha_(self):
        return self.pipeline.named_steps["ridge"].alpha_
