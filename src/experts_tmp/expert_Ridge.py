import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from src.experts.base_expert import BaseExpert

class RidgeExpert(BaseExpert):
    def __init__(
        self,
        features,
        n_split=5,
        alphas = None,
        ):

        # initialisation de la classe mère
        super().__init__(features,name="Ridge")
        # paramètres de la classe
        self.features=features
        self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 50)
        self.cv = TimeSeriesSplit(n_splits=n_split)
        # modèle ridge
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=self.alphas,
                              scoring="neg_mean_squared_error",
                              cv=self.cv))
                              ])

    def fit(self,X,y):
        X_sel = X[self.features]
        self.pipeline.fit(X_sel,y)
        self.is_fitted = True

    def predict(self,X):
        X_sel = X[self.features]
        return self.pipeline.predict(X_sel)
    
    @property
    def best_alpha_(self):
        return self.pipeline.named_steps["ridge"].alpha_
    
    pass