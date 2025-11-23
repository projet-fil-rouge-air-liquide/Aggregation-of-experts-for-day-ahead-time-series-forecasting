import pandas as pd
import numpy as np

Data_clean = pd.read_csv("Processed_data/data_cleaned.csv")

# 1. Calcul de la vitesse du vent à 100 m (hauteur de l'éolienne)
# Calcul de la Vitesse du Vent (Norme) - Aucune modification
Data_clean["Wind_Norm"] = np.sqrt(Data_clean["speed_longitudinale_100m"]**2 + Data_clean["speed_latitudinale_100m"]**2)

# Calcul de l'Angle du Vecteur (vers lequel souffle le vent)
# np.arctan2(V, U) pour obtenir l'angle en radians
Data_clean["Vector_Angle"] = np.arctan2(Data_clean["speed_latitudinale_100m"], Data_clean["speed_longitudinale_100m"])

# Conversion en Degrés
Data_clean["Vector_Angle_Deg"] = Data_clean["Vector_Angle"] * (180 / np.pi)

# Conversion en Direction Météorologique (ajout de 180 degrés)
# La direction météo est l'opposé du vecteur (d'où vient le vent)
# Normalisation pour que l'angle soit entre 0 et 360 degrés
Data_clean["Wind_Dir_Meteo"] = (Data_clean["Vector_Angle_Deg"] + 180) % 360

# créarion de la composante V^3. La puissance d'une éolienne est proportionnelle au cube de la vitesse du vent
Data_clean["Wind_Norm_Cubes"] = Data_clean["Wind_Norm"]**3

# 2. Calcul de la densité de l'air: ρ= n*M/V - ρ= M*P/(R*T)
# Constantes
R = 8.314  # J/(kg·K) 
M = 0.02896  # kg/mol - Masse molaire de l'air sec
Data_clean["Air_density"]=Data_clean["surface_pressure"]*M/(R*Data_clean["2m_temperature"])

# 3. Création des Lags - features historiques/séquencielles
# Ces lags vont modélser l'autocorrélation de la série temporelle de la production éolienne
# c'est à dire la dépendance des valeurs actuelles aux valeurs passées. Dans le cas du vent on peut
# se limiter à des Lag de l'ordre de la semaine. La saisonnalité plus long terme sera capturée par les features calendaires 
# 3.1 construction de tags sur Eolien_MW

Data_clean["Y_lag_1h"] = Data_clean["Eolien_MW"].shift(1)   # Lag 1 heure
Data_clean["Y_lag_24h"] = Data_clean["Eolien_MW"].shift(24)  # Lag 24 heures
Data_clean["Y_lag_48h"] = Data_clean["Eolien_MW"].shift(48)  # Lag 48 heures
Data_clean["Y_lag_72h"] = Data_clean["Eolien_MW"].shift(72)  # Lag 72 heures
Data_clean["Y_lag_96h"] = Data_clean["Eolien_MW"].shift(96)  # Lag 96 heures
Data_clean["Y_lag_120h"] = Data_clean["Eolien_MW"].shift(120)  # Lag 120 heures

# 3.2 construction de lags sur la composante de vent normée
Data_clean["Wind_Norm_lag_1h"] = Data_clean["Wind_Norm"].shift(1)
Data_clean["Wind_Norm_lag_24h"] = Data_clean["Wind_Norm"].shift(24)
Data_clean["Wind_Norm_lag_48h"] = Data_clean["Wind_Norm"].shift(48)
Data_clean["Wind_Norm_lag_72h"] = Data_clean["Wind_Norm"].shift(72)
Data_clean["Wind_Norm_lag_96h"] = Data_clean["Wind_Norm"].shift(96)
Data_clean["Wind_Norm_lag_120h"] = Data_clean["Wind_Norm"].shift(120)

# 4. création des features calendaires. Elles vont modéliser l'impact du facteur saisonnier.
# C'est à dire l'impact du fait qu'on est en janvier (ça souffle => production) ou en juin (moins de production à priori) 
Data_clean["Date_Heure"] = pd.to_datetime(Data_clean["Date_Heure"], errors="coerce")
Data_clean = Data_clean.set_index("Date_Heure",drop=False)

Data_clean["Hour"] = Data_clean.index.hour
Data_clean["Day"] = Data_clean.index.day
Data_clean["Weekday"] = Data_clean.index.weekday
Data_clean["Month"] = Data_clean.index.month

# on transforme les features calendaires en variables cycliques pour mieux capturer la nature périodique du temps
# Heures - Cycle de 24 heures
Data_clean["Hour_sin"] = np.sin(2 * np.pi * Data_clean["Hour"] / 24)
Data_clean["Hour_cos"] = np.cos(2 * np.pi * Data_clean["Hour"] / 24)

# Jour de la semaine - Cycle de 7 jours
Data_clean["Weekday_sin"] = np.sin(2 * np.pi * Data_clean["Weekday"] / 7)
Data_clean["Weekday_cos"] = np.cos(2 * np.pi * Data_clean["Weekday"] / 7)

# Mois - Cycle de 12 mois
Data_clean["Month_sin"] = np.sin(2 * np.pi * Data_clean["Month"] / 12)
Data_clean["Month_cos"] = np.cos(2 * np.pi * Data_clean["Month"] / 12)

# intégrer la courbe de puissance générique d'une éolienne
# on considère une éolienne type avec:
# - une vitesse de cut-in (vitesse minimale pour produire de l'électricité) de 3 m/s
# - une vitesse nominale (vitesse à laquelle l'éolienne produit à pleine capacité) de 12 m/s
# - une vitesse de cut-out (vitesse maximale avant arrêt pour sécurité) de 25 m/s

                                   # A FAIRE


# Clean des dernières données
# supprimer la colonne Perimetre et ajout d'une colonne date en index
Data_clean = Data_clean.drop(columns=["Perimetres"], errors="ignore")
# supprimer les 120 premières lignes avec les valeurs Nan dues aux lags
Data_clean = Data_clean.dropna().reset_index(drop=True)
# vérification des données manquantes
missing_data = Data_clean.isnull().sum()
print("\nDonnées manquantes par colonne :")
print(missing_data[missing_data > 0])       # pas de données manquantes


# sauvegarde des données nettoyées
Data_clean.to_csv("Processed_data/data_engineering.csv", index=False)
