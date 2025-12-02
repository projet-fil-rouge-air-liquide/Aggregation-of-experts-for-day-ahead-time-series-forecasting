from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



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

# expert
expert_ARIMA = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(3,0,2)
)

expert_ARIMA = expert_ARIMA.fit()
y_valid_pred = expert_ARIMA.predict(
    start=train_end,
    end=valid_end - 1,
    exog=X_valid
)


X_future = pd.concat([X_valid, X_test])

y_test_pred = expert_ARIMA.predict(
    start=valid_end,
    end=n_Belgique - 1,
    exog=X_future
)

# RMSE
rmse_expert_ARIMA = np.sqrt(mean_squared_error(y_test,y_test_pred)) 
print("Expert ARIME  :", rmse_expert_ARIMA)


plt.figure(figsize=(7,7))

plt.scatter(y_test, y_test_pred, alpha=0.5, s=10)

plt.xlabel("Valeurs réelles (MW)")
plt.ylabel("Valeurs prédites (MW)")
plt.title("Scatter plot : Réel vs Prédit ARIMA")
max_val = max(max(y_test), max(y_test_pred))   
min_val = min(min(y_test), min(y_test_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.grid(True)
plt.tight_layout()
plt.show()

