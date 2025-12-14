import numpy as np
import pandas as pd

from utils.fonction import fit_predict_eval

from src.config.data_train_valid_test import X_train,X_valid,X_test,y_train,y_valid,y_test
from src.config.features import FEATURES_EN,FEATURES_LGBM,FEATURES_RIDGE
from src.experts import expert_ElasticNet,expert_LGBM,expert_Ridge
from src.agregateurs.agg_lin import AGG_LIN

# création /entrainement des experts
experts = [expert_ElasticNet.ElasticNetExpert(FEATURES_EN),
           expert_LGBM.LGBMExpert(FEATURES_LGBM),
           expert_Ridge.RidgeExpert(FEATURES_RIDGE)]

for exp in experts:
    fit_predict_eval(exp,X_train,X_test,y_train,y_test)
# création des variable de l'agrégateur
X_train_agg =[]
X_test_agg = []
for exp in experts:
    X_train_agg.append(exp.predict(X_valid))    # agrégatino offline
    X_test_agg.append(exp.predict(X_test))      # agrégatino offline
X_train_agg = np.vstack(X_train_agg).T
X_test_agg = np.vstack(X_test_agg).T
# création / entrainement de l'agrégateur linéaire
agg_linalg = AGG_LIN(experts)
fit_predict_eval(agg_linalg,X_train_agg,X_test_agg,y_valid,y_test)
# étude des poids
scaler = agg_linalg.pipeline.named_steps["scaler"]
model = agg_linalg.pipeline.named_steps["model"]

coef_std = model.coef_
coef_real = coef_std / scaler.scale_
intercept_real = model.intercept_ - np.sum(coef_std * scaler.mean_ / scaler.scale_)

print("coef_real:",coef_real)
print("intercept_real:",intercept_real)