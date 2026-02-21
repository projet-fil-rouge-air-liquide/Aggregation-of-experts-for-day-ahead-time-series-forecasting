from lightgbm import LGBMRegressor,early_stopping
from src.experts.base_expert import BaseExpert

class LGBMExpert(BaseExpert):
    def __init__(
        self,
        features,
        n_estimators=700,
        learning_rate=0.05,
        features_name = "unknow",
        num_leaves=31,
        max_depth=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
        ):
        # initialisation de la classe mère
        super().__init__(features,name="LGBM")
        # paramètres spécifiques de la classe LGBM
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.features_name = features_name
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
        self.expert.fit(X_sel,
                        y
        )
        self.is_fitted = True

    def predict(self,X):
        X_sel = X[self.features]
        return self.expert.predict(X_sel)
    
    pass


