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

from src.experts.expert_ElasticNet import ElasticNetExpert
from src.experts.expert_LGBM import LGBMExpert
from src.experts.expert_Ridge import RidgeExpert

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


index = pd.date_range(start='2025-11-28 00:00', periods=72, freq='H')
# sélectionner la fenêtre 72h depuis y_test et réindexer sur l'index choisi
# on utilise les valeurs pour éviter tout conflit d'index existant
y_day_ahead = pd.Series(y_test.iloc[:72].values, index=index)
# convertir les prédictions en Series et aligner les index fournis pour le plot
y_day_ahead_pred_EN = pd.Series(y_pred_exp[0][:72], index=index)
y_day_ahead_pred_R = pd.Series(y_pred_exp[1][:72], index=index)
y_day_ahead_pred_LGBM = pd.Series(y_pred_exp[2][:72], index=index)
y_day_ahead_pred_agg_lin = pd.Series(y_pred_agg[:72], index=index)
# visualisation (tous tracés sur le même index temporel explicite)
plt.figure(figsize=(10,6))
plt.plot(index, y_day_ahead, label="Valeurs réelles", color="red")
plt.plot(index, y_day_ahead_pred_EN, color = "blue", label="Prédictions EN", linestyle='--', linewidth=2, zorder=5)
plt.plot(index, y_day_ahead_pred_R, color="orange", label="Prédictions R", linestyle='--', linewidth=2, zorder=5)
plt.plot(index, y_day_ahead_pred_LGBM, color= "green", label="Prédictions LGBM",linestyle='--', linewidth=2, zorder=5)
plt.plot(index, y_day_ahead_pred_agg_lin, color= "black", label="Prédictions Agg_Lin",linestyle='--', linewidth=2, zorder=5)
plt.xlabel("Index temporel")   
plt.ylabel("Production éolienne (MW)") 
plt.title("Prédictions à 72h - Expert vs Valeurs réelles")
plt.legend()
plt.show()



