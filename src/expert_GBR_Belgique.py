import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor,early_stopping
# Création de deux experts non linéaires avec :
# - expert_2_1: GradientBoostingRegressor
# - expert_2_2: LGBM
import lightgbm


# chargement des données
Data_clean_Belgique = pd.read_csv("../Data/Processed_data/data_engineering_belgique.csv")

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
labels_Belgique = Data_clean_Belgique["Eolien_MW"]

# split des données et features et target - ATTENTION - ne pas mélanger passé et futur
n_Belgique = len(Data_clean_Belgique)
train_end = int(n_Belgique*0.6)
valid_end = int(0.8*n_Belgique)

X_train = Data_clean_Belgique[features_exp1].iloc[:train_end]
y_train = labels_Belgique.iloc[:train_end]

X_valid = Data_clean_Belgique[features_exp1].iloc[train_end:valid_end]  # valid servira à comparer plusieurs modèle (utilisés après)
y_valid = labels_Belgique.iloc[train_end:valid_end]  # valid servira à comparer plusieurs modèle (utilisés après)

X_test = Data_clean_Belgique[features_exp1].iloc[valid_end:]
y_test = labels_Belgique.iloc[valid_end:]

# fit/entrainement du modèle expert 2_1 GradientBoostingRegressor
expert_2_1 = GradientBoostingRegressor(loss="squared_error",     # ne pas standardiser les input dans le cas des algos à arbres: cela peut dégrader les performances
                                      random_state=42,
                                      n_estimators=300,
                                      max_depth=10,
                                      learning_rate=0.05,
                                      subsample=0.8)

expert_2_1.fit(X_train,y_train)                # entrainement
y_pred_2_1 = expert_2_1.predict(X_test)        # prédiction
# performances
rmse_expert_2_1 = np.sqrt(mean_squared_error(y_test,y_pred_2_1)) 
print("Expert 2_1  :", rmse_expert_2_1)

# représentation de la prédiction vs valeurs réeels
plt.figure(figsize=(7,7))

plt.scatter(y_test, y_pred_2_1, alpha=0.5, s=10)

plt.xlabel("Valeurs réelles (MW)")
plt.ylabel("Valeurs prédites (MW)")
plt.title("Scatter plot : Réel vs Prédit GradientBoostingRegressor")
max_val = max(max(y_test), max(y_pred_2_1))   
min_val = min(min(y_test), min(y_pred_2_1))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.grid(True)
plt.tight_layout()
plt.show()
