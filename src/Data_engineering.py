import pandas as pd
import numpy as np

Data_clean = pd.read_csv("Processed_data/data_cleaned.csv")

# 1. Calcul de la vitesse du vent à 100 m (hauteur de l'éolienne)
# Calcul de la Vitesse du Vent (Norme) - Aucune modification
Data_clean["Wind_Norm"] = np.sqrt(Data_clean["u100"]**2 + Data_clean["v100"]**2)

# Calcul de l'Angle du Vecteur (vers lequel souffle le vent)
# Utilise np.arctan2(V, U) pour obtenir l'angle en radians
Data_clean["Vector_Angle"] = np.arctan2(Data_clean["v100"], Data_clean["u100"])

# Conversion en Degrés
Data_clean["Vector_Angle_Deg"] = Data_clean["Vector_Angle"] * (180 / np.pi)

# Conversion en Direction Météorologique (ajout de 180 degrés)
# La direction météo est l'opposé du vecteur (d'où vient le vent)
# Normalisation pour que l'angle soit entre 0 et 360 degrés
Data_clean["Wind_Dir_Meteo"] = (Data_clean["Vector_Angle_Deg"] + 180) % 360


# 2. Calcul de la densité de l'air: ρ= n*M/V - ρ= M*P/(R*T)
# Constantes
R = 8.314  # J/(kg·K) - Constante spécifique pour l'air sec
M = 0.02896  # kg/mol - Masse molaire de l'air sec
Data_clean["Air_density"]=Data_clean["sp"]*M/(R*Data_clean["t2m"])

# 3. Création des Lags - features historiques/séquencielles
# Ces lags vont modélser l'autocorrélation de la série temporelle de la production éolienne
# c'est à dire la dépendance des valeurs actuelles aux valeurs passées
# Lag 1 heure
Data_clean["Y_lag_1h"] = Data_clean["Eolien_MW"].shift(1)
# Lag 1 jour
Data_clean["Y_lag_1j"] = Data_clean["Eolien_MW"].shift(24)
# Lag 1 semaine
Data_clean["Y_lag_7j"] = Data_clean["Eolien_MW"].shift(168)
# Lag 1 mois (30 jours)
Data_clean["Y_lag_30j"] = Data_clean["Eolien_MW"].shift(720)
# Lag 1 saison (3 mois)
Data_clean["Y_lag_season"] = Data_clean["Eolien_MW"].shift(2190)

# 4. création des features calendaires. Elles vont modéliser l'impact du facteur saisonnier.
# C'est à dire l'impact du fait qu'on est en janvier (ça souffle => production) ou en juin (moins de production à priori) 
Data_clean["Date_Heure"] = pd.to_datetime(Data_clean["Date_Heure"], errors="coerce")
Data_clean = Data_clean.set_index("Date_Heure",drop=False)

Data_clean["Hour"] = Data_clean.index.hour
Data_clean["Day"] = Data_clean.index.day
Data_clean["Month"] = Data_clean.index.month

# on transforme les features calendaires en variables cycliques pour mieux capturer la nature périodique du temps
# Heures - Cycle de 24 heures
Data_clean["Hour_sin"] = np.sin(2 * np.pi * Data_clean["Hour"] / 24)
Data_clean["Hour_cos"] = np.cos(2 * np.pi * Data_clean["Hour"] / 24)

# Mois - Cycle de 12 mois
Data_clean["Month_sin"] = np.sin(2 * np.pi * Data_clean["Month"] / 12)
Data_clean["Month_cos"] = np.cos(2 * np.pi * Data_clean["Month"] / 12)


# Clean des dernières données
# supprimer la colonne Perimetre et ajout d'une colonne date en index
Data_clean = Data_clean.drop(columns=["Perimetres"], errors="ignore")
# supprimer les 2190 premières lignes avec les valeurs Nan dues aux lags
Data_clean = Data_clean.dropna().reset_index(drop=True)
# vérification des données manquantes
missing_data = Data_clean.isnull().sum()
print("\nDonnées manquantes par colonne :")
print(missing_data[missing_data > 0])       # pas de données manquantes


# sauvegarde des données nettoyées
Data_clean.to_csv("Processed_data/data_ingineering.csv", index=False)
