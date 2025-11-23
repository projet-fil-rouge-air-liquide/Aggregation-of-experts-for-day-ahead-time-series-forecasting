# chargement des données
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor  
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Data_clean = pd.read_csv("Processed_data/data_engineering.csv")


# création des données de test et d'entrainement
labels = Data_clean["Eolien_MW"]
features = Data_clean[["Wind_Norm_Cubes","Air_density","Wind_Norm","Y_lag_1h","Y_lag_1j","Y_lag_7j","Y_lag_30j","Y_lag_season",
                       "Hour","Day","Month","Hour_sin","Hour_cos","Month_sin","Month_cos"]]
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
expert_1 = Pipeline([
    ('scaler', StandardScaler()),
    ('linreg', LinearRegression())
])

expert_1.fit(X_train, y_train)
y_pred_1 = expert_1.predict(X_test)

# calcul des métriques de performance
rmse_1 = np.sqrt(mean_squared_error(y_test, y_pred_1))
print("Expert 1 - RMSE :", rmse_1)

coefs = expert_1.named_steps["linreg"].coef_
scaled_importance = pd.Series(coefs, index=X_train.columns)

print("\nImportance des variables (normalisées) :")
print(scaled_importance.abs().sort_values(ascending=False))

# Conclusion: la série temporelle est très auto corrélée à l'évènement qui se passe 1h avant. Afin de forcer le modèle à apprendre de la 
# météo, le tag à 1h a été retiré.

# Construction expert 2 - LGBM
# LGBM = Light Gradient Boosting Machine. 
# LGBM fonctionne avec la constrution d'arbres séquentiels:
# - Un premier arbre est créé et fait une première prédiction.
# - Le deuxième arbre créé calcule le résidu de la prédction du premier arbre et tente de corriger cette erreur.
# - Le troisième fait de même avec le résidu du deuxième arbre et ainsi de suite.
# On construit ainsi de manière ittérative une série d'arbres qui corrigent les erreurs des précédents.
# LGBM permet également de capturer les relations non linéaires des features avec notamment:
# - la vitesse du vent au cube
# - l'effet de seuil de la puissance d'une éolienne (0 si vent trop faible ou trop fort et puissance nominale sur une plage de vent)

#  Étape de Correction pour LGBM
# categorical_features = ["Hour", "Day", "Month"]

# for col in categorical_features:
#     # Convertir en type catégorie pour que LGBM ne les traite pas comme des nombres continus
#     X_train[col] = X_train[col].astype('category')
#     X_test[col] = X_test[col].astype('category')

# # entrainement de l'expert 2
# expert_2 = lgb.LGBMRegressor(
#     n_estimators=500,
#     learning_rate=0.03,
#     max_depth=5,
#     min_child_samples=20,
#     objective='regression',
#     subsample=0.8,
#     colsample_bytree=0.8
# )

# expert_2.fit(X_train, y_train)
# y_pred_2 = expert_2.predict(X_test)

# # calcul des métriques de performance
# rmse_2 = np.sqrt(mean_squared_error(y_test, y_pred_2))
# print("Expert 2 - RMSE :", rmse_2)


