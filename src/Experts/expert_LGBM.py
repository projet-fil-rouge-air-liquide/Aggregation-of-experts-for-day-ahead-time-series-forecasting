from lightgbm import LGBMRegressor,early_stopping

class LGBMExpert():
    def __init__(
        self,
        features,
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
        ):

        self.features=features
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample=subsample
        self.colsample_bytree = colsample_bytree
        self.random_state=random_state

        self.expert = LGBMRegressor()
        self.is_fitted=False

    def fit(self,X,y):
        X_sel = X[self.features]
        self.expert.fit(X_sel,y)
        self.is_fitted = True

    def predict(self,X):
        X_sel = X[self.features]
        return self.expert.predict(X_sel)
    
    pass

