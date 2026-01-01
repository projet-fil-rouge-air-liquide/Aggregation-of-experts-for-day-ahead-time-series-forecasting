from sklearn.ensemble import RandomForestRegressor
from src.experts.base_expert import BaseExpert
import warnings
warnings.filterwarnings("ignore")


class RandomForestExpert(BaseExpert):
    def __init__(
        self,
        features,
        n_estimators=600,
        max_depth=15,
        features_name="unknow",
        min_samples_split=10,
        min_samples_leaf=20,
        max_features=0.5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ):
        # initialisation de la classe mère
        super().__init__(features,name="RandomForest")
        # paramètres spécifiques de la classe RandomForest
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features
        self.features_name = features_name
        self.bootstrap=bootstrap
        self.random_state=random_state
        self.n_jobs=n_jobs
        self.expert = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
        )

        self.is_fitted = False

    def fit(self, X, y):
        X_sel = X[self.features]
        self.expert.fit(X_sel, y)
        self.is_fitted = True

    def predict(self, X):
        X_sel = X[self.features]
        return self.expert.predict(X_sel)

