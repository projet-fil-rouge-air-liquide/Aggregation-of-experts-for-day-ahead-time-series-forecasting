import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

from src.experts.base_expert import BaseExpert


class ElasticNetExpert(BaseExpert):
    def __init__(
        self,
        features,
        n_split=5,
        alphas = None,
        max_iter = 10000,
        l1_ratios=None
        ):
        # initialisation de la classe mère
        super().__init__(features,name="ElasticNet")
        # paramètres de modèle
        self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 50)
        self.l1_ratios = l1_ratios if l1_ratios is not None else np.linspace(0.1, 1.0, 10)
        self.cv = TimeSeriesSplit(n_splits=n_split)
        # modèle
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(
                alphas=self.alphas,
                l1_ratio=self.l1_ratios,
                cv=self.cv,
                max_iter=max_iter))
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
        return self.pipeline.named_steps["enet"].alpha_

    @property
    def best_l1_ratio_(self):
        return self.pipeline.named_steps["enet"].l1_ratio_

    def get_coefficients(self):
        enet = self.pipeline.named_steps["enet"]
        return enet.coef_, enet.intercept_
    
    pass