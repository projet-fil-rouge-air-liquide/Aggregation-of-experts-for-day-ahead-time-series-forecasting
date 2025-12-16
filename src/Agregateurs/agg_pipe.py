import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils.fonction import fit_predict_eval

from src.config.data_train_valid_test import X_train,X_valid,X_test,y_train,y_valid,y_test
from src.config.features import FEATURES_EN,FEATURES_LGBM,FEATURES_RIDGE,FEATURES_RDMFOREST
from src.experts import expert_ElasticNet,expert_LGBM,expert_Ridge,expert_RandomForest
from src.agregateurs.agg_lin import AGG_LIN

# création /entrainement des experts
experts = [expert_ElasticNet.ElasticNetExpert(FEATURES_EN),
           expert_LGBM.LGBMExpert(FEATURES_LGBM),
           expert_Ridge.RidgeExpert(FEATURES_RIDGE),
           expert_RandomForest.RandomForestExpert(FEATURES_RDMFOREST)
           ]
y_pred_exp = []
mse_exp = []
for exp in experts:
    y_p,mse_e =fit_predict_eval(exp,X_train,X_test,y_train,y_test)
    y_pred_exp.append(y_p)
    mse_exp.append(mse_e)

for i in mse_exp:
    print(i)

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
y_pred_agg,rmse_agg = fit_predict_eval(agg_linalg,X_train_agg,X_test_agg,y_valid,y_test)
# étude des poids
scaler = agg_linalg.pipeline.named_steps["scaler"]
model = agg_linalg.pipeline.named_steps["model"]

coef_std = model.coef_
coef_real = coef_std / scaler.scale_
intercept_real = model.intercept_ - np.sum(coef_std * scaler.mean_ / scaler.scale_)

print("coef_real:",coef_real)
print("intercept_real:",intercept_real)


index = pd.date_range(start='2025-11-11 00:00', end='2025-11-23 00:00', freq='H')
n=len(index)
y_day_ahead = pd.Series(y_test.iloc[:n].values, index=index)
# convertir les prédictions en Series et aligner les index fournis pour le plot
y_day_ahead_pred_EN = pd.Series(y_pred_exp[0][:n], index=index)
y_day_ahead_pred_R = pd.Series(y_pred_exp[1][:n], index=index)
y_day_ahead_pred_LGBM = pd.Series(y_pred_exp[2][:n], index=index)
y_day_ahead_pred_agg_lin = pd.Series(y_pred_agg[:n], index=index)
# visualisation (tous tracés sur le même index temporel explicite)

plt.figure(figsize=(10,6))
plt.plot(index, y_day_ahead, label="Valeurs réelles", color="red")
plt.plot(index, y_day_ahead_pred_EN, color = "blue", label="Prédictions EN", linestyle='--', linewidth=2, zorder=5)
plt.plot(index, y_day_ahead_pred_R, color="orange", label="Prédictions R", linestyle='--', linewidth=2, zorder=5)
plt.plot(index, y_day_ahead_pred_LGBM, color= "green", label="Prédictions LGBM",linestyle='--', linewidth=2, zorder=5)
plt.plot(index, y_day_ahead_pred_agg_lin, color= "black", label="Prédictions Agg_Lin",linestyle='--', linewidth=2, zorder=5)
plt.xlabel("Index temporel")   
plt.ylabel("Production éolienne (MW)") 
plt.title("Prédictions - Expert vs Valeurs réelles")
plt.legend()
plt.show()


# représentation de la prédiction vs valeurs réeels
plt.style.use("ggplot")

plt.figure(figsize=(9,7))

plt.hexbin(
    y_test, y_pred_agg,
    gridsize=60,
    cmap="viridis",
    mincnt=1,
    norm=LogNorm()
)
plt.colorbar(label="Densité (log)")

max_val = max(max(y_test), max(y_pred_agg))
min_val = min(min(y_test), min(y_pred_agg))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

coef = np.polyfit(y_test, y_pred_agg, 1)
poly = np.poly1d(coef)
plt.plot(y_test, poly(y_test), color="orange", linewidth=2)

plt.xlabel("Valeurs réelles (MW)")
plt.ylabel("Valeurs prédites (MW)")
plt.title("Réel vs Prédit — EN")

plt.tight_layout()
plt.show()