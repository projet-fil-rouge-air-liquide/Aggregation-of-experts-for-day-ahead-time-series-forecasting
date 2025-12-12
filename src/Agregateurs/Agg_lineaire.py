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

from src.Experts.expert_ElasticNet import ElasticNetExpert
from src.Experts.expert_LGBM import LGBMExpert
from src.Experts.expert_Ridge import RidgeExpert

# créer Experts
expert_EN = ElasticNetExpert(features=FEATURES_EN)
expert_LGBM = LGBMExpert(features=FEATURES_LGBM)
expert_Ridge = RidgeExpert(features=FEATURES_RIDGE)
experts = [expert_EN,expert_LGBM,expert_Ridge]
# fit/predict/eval
for exp in experts:
    fit_predict_eval(exp,X_train,X_test,y_train,y_test)
# création des varaibles d'entrainement des agragéteurs
X_train_exp =[]
X_test_exp = []
for exp in experts:
    X_train_exp.append(exp.predict(X_valid))
    X_test_exp.append(exp.predict(X_test))
X_train_exp = np.vstack(X_train_exp).T
X_test_exp = np.vstack(X_test_exp).T
# création de l'agrégateur
agg = Pipeline([
    ("scale", StandardScaler()),
    ("régression",LinearRegression())])
# entrainement de l'égrégateur
fit_predict_eval(agg,X_train_exp,X_test_exp,y_valid,y_test)


# index = pd.date_range(start='2025-12-01 00:00', periods=24, freq='H')
# # sélectionner la fenêtre 24h depuis y_test et réindexer sur l'index choisi
# # on utilise les valeurs pour éviter tout conflit d'index existant
# y_day_ahead = pd.Series(y_test.iloc[:24].values, index=index)
# # convertir les prédictions en Series et aligner les index fournis pour le plot
# y_day_ahead_pred_EN = pd.Series(y_pred_1[:24], index=index)
# y_day_ahead_pred_R = pd.Series(y_pred_12[:24], index=index)
# y_day_ahead_pred_LGBM = pd.Series(y_pred_2[:24], index=index)
# # visualisation (tous tracés sur le même index temporel explicite)
# plt.figure(figsize=(10,6))
# plt.plot(index, y_day_ahead, label="Valeurs réelles", color="red", marker='o')
# plt.plot(index, y_day_ahead_pred_EN, color = "blue", label="Prédictions EN", marker='x', linestyle='--', linewidth=2, zorder=5)
# # couleur et style Ridge modifiés pour meilleure lisibilité
# plt.plot(index, y_day_ahead_pred_R, color="orange", label="Prédictions R", marker='s', linestyle='-.', linewidth=2)
# plt.plot(index, y_day_ahead_pred_LGBM, color= "green", label="Prédictions LGBM", marker='x')
# plt.xlabel("Index temporel")   
# plt.ylabel("Production éolienne (MW)") 
# plt.title("Prédictions à 24h - Expert vs Valeurs réelles")
# plt.legend()
# plt.show()



