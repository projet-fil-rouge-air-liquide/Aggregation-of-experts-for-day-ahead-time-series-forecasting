import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet,ElasticNetCV,LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.colors import LogNorm

from src.config.features import FEATURES_EN, FEATURES_RIDGE, FEATURES_LGBM
from src.config.data_train_valid_test import X_train,X_valid,X_test,y_train,y_valid,y_test 
from utils.fonction import fit_predict_eval,rmse,predict,fit

from src import Experts
from src.Experts.expert_LGBM import LGBMExpert
from src.Experts.expert_Ridge import RidgeExpert

class agg_lin():
    def __init__(self,experts):
        self.experts = experts
        self.experts_names = [e.names for e in experts] 

        pass

# créer Experts
expert_EN = ElasticNetExpert(features=FEATURES_EN)
expert_LGBM = LGBMExpert(features=FEATURES_LGBM)
expert_Ridge = RidgeExpert(features=FEATURES_RIDGE)
experts = [expert_EN,expert_Ridge,expert_LGBM]
# fit/predict/eval des experts
y_pred_exp = []
rmse_exp = []
for exp in experts:
    y_pred,rmse_e = fit_predict_eval(exp,X_train,X_test,y_train,y_test)
    y_pred_exp.append(y_pred)
    rmse_exp.append(rmse_e)
# création des varaibles d'entrainement des agragéteurs
X_train_agg =[]
X_test_agg = []
for exp in experts:
    X_train_agg.append(exp.predict(X_valid))
    X_test_agg.append(exp.predict(X_test))
X_train_exp = np.vstack(X_train_agg).T
X_test_exp = np.vstack(X_test_agg).T
# création de l'agrégateur
agg_lin = Pipeline([
    ("scale", StandardScaler()),
    ("régression",LinearRegression())])
# entrainement de l'égrégateur
y_pred_agg,rmse_agg = fit_predict_eval(agg_lin,X_train_exp,X_test_exp,y_valid,y_test)






