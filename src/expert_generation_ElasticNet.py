import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

# chargement des données
Data_clean = pd.read_csv("../Data/Processed_data/data_engineering.csv")

# sélection des features pour l'expert 1 (toutes les features de Data_engineering)
features_exp1 = [
    "speed_longitudinale_100m",
    "speed_latitudinale_100m",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "surface_pressure",
    "Wind_Norm",
    "Wind_Norm_Cubes",
    "wind_std_3h",
    "wind_cv_3h",
    "Y_lag_1h",
    "Y_lag_24h",
    "Wind_Norm_lag_1h",
    "Wind_Norm_lag_24h",
    "Hour_sin",
    "Hour_cos",
    "Weekday_sin",
    "Weekday_cos",
    "Month_sin",
    "Month_cos",
    "P_curve",
    "Wind_mean_3h",
    "Air_density",
    "Wind_Dir_Meteo_sin",
    "Wind_Dir_Meteo_cos"
]
# définition du Label
labels = Data_clean["Eolien_MW"]
# split des données et features et target - ATTENTION - ne pas mélanger passé et futur
n = len(Data_clean)
train_end = int(n*0.6)
valid_end = int(0.8*n)

X_train = Data_clean[features_exp1].iloc[:train_end]
y_train = labels.iloc[:train_end]

X_valid = Data_clean[features_exp1].iloc[train_end:valid_end]  # valid servira à comparer plusieurs modèle (utilisés après)
y_valid = labels.iloc[train_end:valid_end]  # valid servira à comparer plusieurs modèle (utilisés après) 

X_test = Data_clean[features_exp1].iloc[valid_end:]
y_test = labels.iloc[valid_end:]

# K_fold pour estimer le meilleur hyper paramètre
tsv = TimeSeriesSplit(n_splits=5) # Kfold appliqué aux séries temporelles
lambdas = np.logspace(-3,3,50)
l1s=np.linspace(0.1, 1.0, 10)
expert_1_EN_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNetCV(alphas=lambdas,
                           l1_ratio=l1s,
                           cv=tsv,
                           max_iter=10000))
])

expert_1_EN_cv.fit(X_train,y_train)  # entrainement sur train
best_alpha = expert_1_EN_cv.named_steps["enet"].alpha_
best_l1 = expert_1_EN_cv.named_steps["enet"].l1_ratio_

print("Alpha optimal (lambda) :", best_alpha)
print("l1_ratio optimal (gamma) :", best_l1)

y_pred_1 = expert_1_EN_cv.predict(X_test) # prédiction

# estimation des performances de l'expert 1_EN
rmse_expert1_EN = np.sqrt(mean_squared_error(y_test, y_pred_1))
print("Expert 1 EN - RMSE :", rmse_expert1_EN)

# estimation des coefficients
coeff_EN = expert_1_EN_cv.named_steps["enet"].coef_
intercept_EN = expert_1_EN_cv.named_steps["enet"].intercept_

coef_table = pd.DataFrame({
    "coefficients":coeff_EN,
    "features":features_exp1
}).sort_values("coefficients",ascending=True)

print(coef_table)

# représentation de la prédiction vs valeurs réeels
plt.figure(figsize=(7,7))

plt.scatter(y_test, y_pred_1, alpha=0.5, s=10)

plt.xlabel("Valeurs réelles (MW)")
plt.ylabel("Valeurs prédites (MW)")
plt.title("Scatter plot : Réel vs Prédit (Elastic_Net)")

max_val = max(max(y_test), max(y_pred_1))   # Diagonale parfaite
min_val = min(min(y_test), min(y_pred_1))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.grid(True)
plt.tight_layout()
plt.show()


# On obtient des coefficients nuls pour:
# Wind_Norm_lag_1h
# Air_density -> ok: colinéarité avec la pression et la température
# Month_sin > ok: colinéarité avec Month_cos
# Weekday_cos -> ok: colinéarité avec Weekday_sins


