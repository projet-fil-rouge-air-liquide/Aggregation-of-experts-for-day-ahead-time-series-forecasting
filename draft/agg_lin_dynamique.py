import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error

from utils.fonction import fit_predict_eval

from src.config.data_train_valid_test import X_train,X_valid,X_test,y_train,y_valid,y_test
from src.config.features import FEATURES_EN,FEATURES_LGBM,FEATURES_RIDGE,FEATURES_RDMFOREST
from src.experts import expert_ElasticNet,expert_LGBM,expert_Ridge,expert_RandomForest
from src.agregateurs.agg_lin import AGG_LIN

# création /entrainement des experts
experts = [expert_ElasticNet.ElasticNetExpert(FEATURES_EN),
           #expert_LGBM.LGBMExpert(FEATURES_LGBM),
           expert_Ridge.RidgeExpert(FEATURES_RIDGE),
           expert_RandomForest.RandomForestExpert(FEATURES_RDMFOREST)
           ]
y_pred_exp = []
mse_exp = []

# initialisation des paramètres
T = len(X_test)
gamma = 0.5
eta = 0.5
K = len(experts)
w = np.ones(K)/K
# stocker les prédictions et stocker les poids
all_y_pred = []
all_w = []

for exp in experts:
    fit_predict_eval(exp,X_train,X_test,y_train,y_test)

# for t in range(T):
#     y_true_t = y_test.iloc[t]
#     # prédiction
#     y_pred_exp = np.array([exp.predict(X_test.iloc[[t]]) for exp in experts]).flatten()
#     # prédiction agrégée
#     y_pred_exp_agg = np.dot(w,y_pred_exp)
#     # stocker les poids avant mise à jour
#     all_w.append(w.copy())
#     # stocker les prédiction agrégée
#     all_y_pred.append(y_pred_exp_agg)
#     # calcul de la perte pour chaque expert agrégé
#     loss_exp_t = np.abs(y_pred_exp - y_true_t)
#     # mise à jour des poids pour chaque expert à chaque instant t
#     w = w*np.exp(-eta*loss_exp_t)
#     w = w/np.sum(w)  # on normalise les poids en les sommants à chaque instant
# # stocker les poids à chaque instant pour chaque experts
# all_w.append(w)


# --- 1. PRÉ-CALCUL : Les experts prédisent tout le bloc de test ---
print("Pré-calcul des prédictions des experts...")
# On crée une matrice de taille (T, K) -> (nb_points, 3 experts)
preds_matrix = []
for exp in experts:
    preds_matrix.append(exp.predict(X_test))
preds_matrix = np.array(preds_matrix).T  # Transpose pour avoir [t, expert]

# --- 2. CONVERSION DES CIBLES ---
y_true_values = y_test.values.flatten()
T = len(y_true_values)

# --- 3. BOUCLE D'AGRÉGATION (Ultra-rapide) ---
print("Lancement de l'agrégation dynamique...")
for t in range(T):
    # On récupère les prédictions déjà calculées pour l'instant t
    y_p_t = preds_matrix[t]  # Vecteur de taille K
    y_true_t = y_true_values[t]
    
    # Agrégation
    y_pred_agg_t = np.dot(w, y_p_t)
    
    all_w.append(w.copy())
    all_y_pred.append(y_pred_agg_t)
    
    # Mise à jour mathématique
    loss_exp_t = np.abs(y_p_t - y_true_t)
    w = w * np.exp(-eta * loss_exp_t)
    w /= np.sum(w)
    
    # Fixed Share
    w = (1 - gamma) * w + (gamma / K)

print("Terminé !")

from sklearn.metrics import mean_squared_error

# Conversion en array si ce n'est pas déjà fait
# --- APRÈS LA BOUCLE ---

# 1. Conversion de la liste en Série Pandas (pour garder l'index temporel)
all_y_pred = pd.Series(all_y_pred, index=y_test.index, name="Pred_Agg")

# 2. Calcul de la MSE (façon Pandas)
# On calcule l'erreur au carré pour chaque point, puis on fait la moyenne
mse_agg = ((y_test - all_y_pred)**2).mean()
print(f"MSE de l'agrégateur dynamique : {mse_agg:.2f}")

# 3. Correction du graphique (maintenant .min() fonctionnera sur la Série)
min_val = min(y_test.min(), all_y_pred.min())
max_val = max(y_test.max(), all_y_pred.max())

# ... reste de votre code de représentation graphique ...
# for i in mse_exp:
#     print("Valeur de MAE:",i)

# # création des variable de l'agrégateur
# X_train_agg =[]
# X_test_agg = []
# for exp in experts:
#     X_train_agg.append(exp.predict(X_valid))    # agrégatino offline
#     X_test_agg.append(exp.predict(X_test))      # agrégatino offline
# X_train_agg = np.vstack(X_train_agg).T
# X_test_agg = np.vstack(X_test_agg).T

# # création / entrainement de l'agrégateur linéaire
# agg_linalg = AGG_LIN(experts)
# y_pred_agg,rmse_agg = fit_predict_eval(agg_linalg,X_train_agg,X_test_agg,y_valid,y_test)
# # étude des poids
# scaler = agg_linalg.pipeline.named_steps["scaler"]
# model = agg_linalg.pipeline.named_steps["model"]

# coef_std = model.coef_
# coef_real = coef_std / scaler.scale_
# intercept_real = model.intercept_ - np.sum(coef_std * scaler.mean_ / scaler.scale_)

# print("coef_real:",coef_real)
# print("intercept_real:",intercept_real)
# print("MAE de l'agg linéaire",rmse_agg)


# index = pd.date_range(start='2025-11-11 00:00', end='2025-11-23 00:00', freq='H')
# n=len(index)
# y_day_ahead = pd.Series(y_test.iloc[:n].values, index=index)
# # convertir les prédictions en Series et aligner les index fournis pour le plot
# y_day_ahead_pred_EN = pd.Series(y_pred_exp[0][:n], index=index)
# y_day_ahead_pred_R = pd.Series(y_pred_exp[1][:n], index=index)
# y_day_ahead_pred_LGBM = pd.Series(y_pred_exp[2][:n], index=index)
# y_day_ahead_pred_agg_lin = pd.Series(y_pred_agg[:n], index=index)
# # visualisation (tous tracés sur le même index temporel explicite)

# plt.figure(figsize=(10,6))
# plt.plot(index, y_day_ahead, label="Valeurs réelles", color="red")
# plt.plot(index, y_day_ahead_pred_EN, color = "blue", label="Prédictions EN", linestyle='--', linewidth=2, zorder=5)
# plt.plot(index, y_day_ahead_pred_R, color="orange", label="Prédictions R", linestyle='--', linewidth=2, zorder=5)
# plt.plot(index, y_day_ahead_pred_LGBM, color= "green", label="Prédictions LGBM",linestyle='--', linewidth=2, zorder=5)
# plt.plot(index, y_day_ahead_pred_agg_lin, color= "black", label="Prédictions Agg_Lin",linestyle='--', linewidth=2, zorder=5)
# plt.xlabel("Index temporel")   
# plt.ylabel("Production éolienne (MW)") 
# plt.title("Prédictions - Expert vs Valeurs réelles")
# plt.legend()
# plt.show()


# # représentation de la prédiction vs valeurs réeels
# plt.style.use("ggplot")

# plt.figure(figsize=(9,7))

# plt.hexbin(
#     y_test, y_pred_agg,
#     gridsize=60,
#     cmap="viridis",
#     mincnt=1,
#     norm=LogNorm()
# )
# plt.colorbar(label="Densité (log)")

# max_val = max(max(y_test), max(y_pred_agg))
# min_val = min(min(y_test), min(y_pred_agg))
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

# coef = np.polyfit(y_test, y_pred_agg, 1)
# mae = mean_absolute_error(y_test,y_pred_agg)
# poly = np.poly1d(coef)
# plt.plot(y_test, poly(y_test), color="orange", linewidth=2)
# # plt.plot(
# #     [min_val, max_val], 
# #     [min_val + mae, max_val + mae], 
# #     color="black", 
# #     linestyle=':', 
# #     linewidth=1.5, 
# #     label=f"+/- MAE ({mae:.2f})"
# # )
# # plt.plot(
# #     [min_val, max_val], 
# #     [min_val - mae, max_val - mae], 
# #     color="black", 
# #     linestyle=':', 
# #     linewidth=1.5
# # )


# plt.xlabel("Valeurs réelles (MW)")
# plt.ylabel("Valeurs prédites (MW)")
# plt.title("Réel vs Prédit — AGG Linéaire")

# plt.tight_layout()
# plt.show()

