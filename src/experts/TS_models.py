import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# représenter la série temporelle sur 24h à partir de données test
# créer index temporel factice (24 pas horaires)
index = pd.date_range(start='2025-12-01 00:00', periods=24, freq='H')
# sélectionner la fenêtre 24h depuis y_test et réindexer sur l'index choisi
# on utilise les valeurs pour éviter tout conflit d'index existant
y_day_ahead = pd.Series(y_test.iloc[:24].values, index=index)
# convertir les prédictions en Series et aligner les index fournis pour le plot
y_day_ahead_pred_EN = pd.Series(y_pred_1[:24], index=index)
y_day_ahead_pred_R = pd.Series(y_pred_12[:24], index=index)
y_day_ahead_pred_LGBM = pd.Series(y_pred_2[:24], index=index)
# visualisation (tous tracés sur le même index temporel explicite)
plt.figure(figsize=(10,6))
plt.plot(index, y_day_ahead, label="Valeurs réelles", color="red", marker='o')
plt.plot(index, y_day_ahead_pred_EN, color = "blue", label="Prédictions EN", marker='x', linestyle='--', linewidth=2, zorder=5)
# couleur et style Ridge modifiés pour meilleure lisibilité
plt.plot(index, y_day_ahead_pred_R, color="orange", label="Prédictions R", marker='s', linestyle='-.', linewidth=2)
plt.plot(index, y_day_ahead_pred_LGBM, color= "green", label="Prédictions LGBM", marker='x')
plt.xlabel("Index temporel")   
plt.ylabel("Production éolienne (MW)") 
plt.title("Prédictions à 24h - Expert vs Valeurs réelles")
plt.legend()
plt.show()