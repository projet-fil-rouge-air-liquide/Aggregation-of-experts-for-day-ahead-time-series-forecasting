import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor

from utils.fonction import fit_predict_eval, calculate_nmae, predict_eval

from src.config.data_train_valid_test import X_train,X_valid,X_test,y_train,y_valid,y_test
from src.config.data_train_valid_test_to_24 import X_train_24,X_valid_24,X_test_24,y_train_24,y_valid_24,y_test_24

from src.config.features import features_groupe
from src.experts import expert_ElasticNet,expert_LGBM,expert_Ridge,expert_RandomForest
from src.agregateurs.agg_lin import AGG_LIN


# classes d'expert
expert_classe = [expert_Ridge.RidgeExpert,
                 expert_RandomForest.RandomForestExpert,
                 expert_LGBM.LGBMExpert
                 ]

experts=[]
for name,features in features_groupe.items():
    for exp in expert_classe:
        experts.append(exp(features=features,features_name=name))

results=[]
experts_preds_val =[]
experts_preds_test = []

ref_capacity = max(y_train.max(), y_valid.max(), y_test.max())

for exp in experts:
    exp.fit(X_train,y_train)

    y_pred, wape_value, mae, mse, nmae_value = predict_eval(
        exp, 
        X_test, 
        y_test, 
        capacity=ref_capacity
    ) 

# construction des variables pour l'agrégation
    # variables validation
    y_pred_val = exp.predict(X_valid) 
    experts_preds_val.append(y_pred_val)
    # variables test
    experts_preds_test.append(y_pred.flatten())

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
y_valid_flat = y_valid.values.flatten()
X_val_agg = np.column_stack(experts_preds_val)
X_test_agg = np.column_stack(experts_preds_test)

# ---------------- création de l'agrégateur linéaire avec cross validation - RIDGE ---------------- #
tsv = TimeSeriesSplit(n_splits=5)
alpha_test = np.logspace(2,10,20)
agg_ridge_cv = RidgeCV(alphas=alpha_test,
                       cv=tsv,
                       scoring='neg_mean_absolute_error',
                       fit_intercept=False)
agg_ridge_cv.fit(X_val_agg,y_valid_flat)

alpha_best = agg_ridge_cv.alpha_
print('best alpha: ',alpha_best)

y_pred_agg_test, wape_agg, mae_agg, mse_agg, nmae_agg = predict_eval(
    agg_ridge_cv,
    X_test_agg,
    y_test,
    capacity=ref_capacity
)
# récupération des coefficients
coefficient = agg_ridge_cv.coef_        
intercept = agg_ridge_cv.intercept_

df_coef = pd.DataFrame({
    'coefficient':coefficient,
    'intercept':intercept,
    "expert": [f"{exp.name}_{exp.features_name}" for exp in experts]
})
#print(df_coef)
print(f"nMAE de l'agrégateur Ridge cv : {nmae_agg:.2f}%")

# ---------------- création de l'agrégateur linéaire avec cross validation - LASSO ---------------- #
tsv_l = TimeSeriesSplit(5)
alpha_test = np.logspace(-4,4,20)
agg_lasso_cv = LassoCV(
    alphas=alpha_test,
    cv=tsv,
    random_state=0,
    max_iter=10000,
    fit_intercept=False,
    selection='random',
    n_jobs=-1,
)

agg_lasso_cv.fit(X_val_agg,y_valid_flat)

alpha_best = agg_lasso_cv.alpha_
print('best alpha: ',alpha_best)

y_pred_agg_test, wape_agg, mae_agg_la, mse_agg, nmae_agg_la = predict_eval(
    agg_lasso_cv,
    X_test_agg,
    y_test,
    capacity=ref_capacity
)
coef_lasso = agg_lasso_cv.coef_
experts_conserves = np.sum(coef_lasso != 0)
print(f"Meilleur alpha : {agg_lasso_cv.alpha_}")
print(f"Nombre d'experts conservés par le Lasso : {experts_conserves} / {len(coef_lasso)}")
print(f"nMAE de l'agrégateur Lasso : {nmae_agg_la:.2f}%")
# ---------------- création de l'agrégateur non linéaire  ---------------- #

agg_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=10,
    random_state=10,
    n_jobs=-1
)
agg_rf.fit(X_val_agg,y_valid_flat)

y_pred_agg_test, wape_agg_rf, mae_agg_rf, mse_agg_rf, nmae_agg_rf = predict_eval(
    agg_rf,
    X_test_agg,
    y_test,
    capacity=ref_capacity
)
print(f"nMAE de l'agrégateur RF : {nmae_agg_rf:.2f}%")


import matplotlib.pyplot as plt
import numpy as np

def plot_day_ahead(day_index, y_test_24, experts_preds_test, expert_names, y_agg_flat, ref_capacity):
    plt.figure(figsize=(12, 6))
    
    # 1. Tracé du Réel
    real_day = y_test_24[day_index]
    plt.plot(range(24), real_day, label="Réel (Production)", color='black', linewidth=3, zorder=10)
    
    # 2. Tracé des meilleurs experts (en pointillés)
    for i in range(min(3, len(experts_preds_test))):
        pred_matrix = experts_preds_test[i].reshape(-1, 24)
        plt.plot(range(24), pred_matrix[day_index], '--', label=f"Exp: {expert_names[i]}", alpha=0.6)

    # 3. Tracé de l'AGRÉGATEUR 
    #agg_matrix = y_agg_flat.reshape(-1, 24)
    #plt.plot(range(24), agg_matrix[day_index], color='red', linewidth=2.5, label="AGRÉGATEUR FINAL", zorder=11)

    # Cosmétique
    plt.title(f"Profil Day-Ahead - Jour")# {day_index} (nMAE Agg: {nmae_agg:.2f}%)")
    plt.xlabel("Heure")
    plt.ylabel("Puissance (MW)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# On récupère les noms des experts triés par performance
sorted_names = results.sort_values("nmae_%")["Exp_name"].tolist()
# On récupère les prédictions correspondantes
sorted_preds = [experts_preds_test[i] for i in results.sort_values("nmae_%").index]

# Tracer le jour 10 par exemple
plot_day_ahead(28, y_test_24, sorted_preds, sorted_names, y_pred_agg_test, ref_capacity)

