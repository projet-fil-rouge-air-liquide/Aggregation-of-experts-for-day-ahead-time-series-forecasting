import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error

from utils.fonction import fit_predict_eval, calculate_nmae

from src.config.data_train_valid_test import X_train,X_valid,X_test,y_train,y_valid,y_test
from src.config.data_train_valid_test_to_24 import X_train_24,X_valid_24,X_test_24,y_train_24,y_valid_24,y_test_24

from src.config.features import features_groupe
from src.experts import expert_ElasticNet,expert_LGBM,expert_Ridge,expert_RandomForest
from src.agregateurs.agg_lin import AGG_LIN


# classes d'expert
expert_classe = [expert_Ridge.RidgeExpert,
                 expert_RandomForest.RandomForestExpert,
                 #expert_LGBM.LGBMExpert
                 ]

experts=[]
for name,features in features_groupe.items():
    for exp in expert_classe:
        experts.append(exp(features=features,features_name=name))

results=[]
experts_preds_val =[]
experts_preds_test = []

y_test_flat = y_test_24.flatten()
ref_capacity = y_test_flat.max()

for exp in experts:
    y_pred,wape_value,mae,mse =fit_predict_eval(exp,X_train_24,X_test_24,y_train_24,y_test_24)

    y_pred_flat = y_pred.flatten()
    nmae_value = calculate_nmae(y_test_flat, y_pred_flat, capacity=ref_capacity)

# construction des variables pour l'agrégation
    # variables validation
    y_pred_val = exp.predict(X_valid_24) 
    experts_preds_val.append(y_pred_val)
    # variables test
    experts_preds_test.append(y_pred)

    results.append({
        "Exp_name": f"{exp.name}_{exp.features_name}",
        "nmae_%": nmae_value,
        "wape": wape_value,
        "mae": mae,
        "mse": mse
})
    
results = pd.DataFrame(results)
results.sort_values(ascending=True,by="nmae_%",inplace=True)
print(results)

# conditionnemnet des variables d'agrégation avec np.column_stack
X_val_agg = np.column_stack(experts_preds_val)
X_test_agg = np.column_stack(experts_preds_test)

# import numpy as np
# from opera import Mixture

# print("--- Diagnostic OPERA ---")
# source = inspect.getsource(opera.Mixture.__init__)
# print(source)

# y_vals = y_valid.values.flatten()

# # 2. Transformation de tes experts en DataFrame (C'est la ligne cruciale)
# if isinstance(X_val_agg, np.ndarray):
#     # On transforme l'array en DataFrame
#     exp_vals = pd.DataFrame(X_val_agg)
# else:
#     exp_vals = X_val_agg

# # 3. Appel à Mixture (avec loss_type='ls' ou sans l'argument)
# mix = Mixture(
#     y=y_vals, 
#     experts=exp_vals, 
#     model='ewa'
# )

# # 4. Pour la prédiction, fais de même pour le test
# X_test_df = pd.DataFrame(X_test_agg)
# y_pred_flat = mix.predict(experts=X_test_df)




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
# #y_day_ahead_pred_LGBM = pd.Series(y_pred_exp[2][:n], index=index)
# y_day_ahead_pred_agg_lin = pd.Series(y_pred_agg[:n], index=index)
# # visualisation (tous tracés sur le même index temporel explicite)

# plt.figure(figsize=(10,6))
# plt.plot(index, y_day_ahead, label="Valeurs réelles", color="red")
# plt.plot(index, y_day_ahead_pred_EN, color = "blue", label="Prédictions EN", linestyle='--', linewidth=2, zorder=5)
# plt.plot(index, y_day_ahead_pred_R, color="orange", label="Prédictions R", linestyle='--', linewidth=2, zorder=5)
# #plt.plot(index, y_day_ahead_pred_LGBM, color= "green", label="Prédictions LGBM",linestyle='--', linewidth=2, zorder=5)
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

# # droite idéale y = x
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

# # régression linéaire
# coef = np.polyfit(y_test, y_pred_agg, 1)
# poly = np.poly1d(coef)

# x_sorted = np.sort(y_test)
# plt.plot(x_sorted, poly(x_sorted), color="orange", linewidth=2)

# plt.axis("equal")
# plt.xlim(min_val, max_val)
# plt.ylim(min_val, max_val)

# plt.xlabel("Valeurs réelles (MW)")
# plt.ylabel("Valeurs prédites (MW)")
# plt.title("Réel vs Prédit — AGG Linéaire")

# plt.tight_layout()
# plt.show()
# %%
