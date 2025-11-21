# chargement des données
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

Data_clean = pd.read_csv("Processed_data/data_engineering.csv")
print(Data_clean.columns.tolist())
# création des données de test et d'entrainement
labels = Data_clean["Eolien_MW"]
features = Data_clean[["Wind_Norm_Cubes","Air_density","Wind_Norm","Y_lag_1h","Y_lag_1j","Y_lag_7j","Y_lag_30j","Y_lag_season",
                       "Hour","Day","Month","Hour_sin","Hour_cos","Month_sin"]]
# split des données et features et target - ATTENTION - ne pas mélanger passé et futur
X_train, X_test, y_train, y_test = train_test_split(features, labels,test_size=0.2,shuffle=False)   # pour une série temporelle, on ne peut pas mélanger passée et futur. shuffle empêche ce mélange.

# Construction expert 1 - modèle de régression linéaire simple
# On a: P= 1/2 ⋅ρ⋅A⋅V^3 -> puissance d'une éolienne
# On en déduit le choix des valeurs explicatives:
#  - la vitesse du vent à 100m
#  - la vitesse du vent au cube à 100m. En effet, la puissance d'une éolienne est proportionnelle au cube de la vitesse du vent
#  - la densité de l'air 
#  - A : la surface de l'éolienne (non prise en compte dans le modèle)
#  - Y_lag - modéléliser l'autocorrélation de de la série temporelle de la production de l'éolienne
#  - Hour/Day/Month/Hour_sin/Hour_cos/Month_sin/Month_cos: modéliser la saisonalité dans le temps

# entrainement de l'expert 1
expert_1 = linear_model.LinearRegression()
expert_1.fit(X_train, y_train)
y_pred_1 = expert_1.predict(X_test)

# calcul des métriques de performance
mse_1 = mean_squared_error(y_test, y_pred_1)
print("Expert 1 - MSE :", mse_1)


