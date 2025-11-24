import pandas as pd
import numpy as np

Data_clean = pd.read_csv("Processed_data/data_cleaned.csv")

# 1. Feauture engineering sur la composante vent
# 1.1 Calcul de la vitesse du vent à 100 m (hauteur de l'éolienne)
# Calcul de la Vitesse du Vent (Norme) - Aucune modification
Data_clean["Wind_Norm"] = np.sqrt(Data_clean["speed_longitudinale_100m"]**2 + Data_clean["speed_latitudinale_100m"]**2)

# créarion de la composante V^3 car la puissance d'une éolienne est proportionnelle au cube de la vitesse du vent
Data_clean["Wind_Norm_Cubes"] = Data_clean["Wind_Norm"]**3

# 1.2 Mesure de la variabilité du vent et indice de turbulence: en effet, la production éolienne est affectée par la variabilité du vent
Data_clean["Wind_mean_3h"] = Data_clean['Wind_Norm'].rolling(window=3, min_periods=1).mean() # mesure le vent moyen sur 3 heures
Data_clean['wind_std_3h'] = Data_clean['Wind_Norm'].rolling(window=3, min_periods=1).std() # mesure l'écart type du vent sur 3 heures
Data_clean['wind_cv_3h'] = Data_clean['wind_std_3h'] / Data_clean['Wind_mean_3h'].replace(0, np.nan) # mesure la turbulance comme le coefficient de variation
Data_clean['wind_cv_3h'] = Data_clean['wind_cv_3h'].fillna(0) # retirer toutes les valeurs NaN résultant de la division par zéro

# 1.3 intégration du caractère circulaire du vent
# calcul de la direction du vent
Data_clean["Wind_Direction"] = np.arctan2(Data_clean["speed_latitudinale_100m"], Data_clean["speed_longitudinale_100m"])
# conversion en degrés
Data_clean["Wind_Dir_Meteo"] = (np.degrees(Data_clean["Wind_Direction"]) + 360) % 360
# passer à la direction météo (d'où vient le vent)
Data_clean["Wind_Dir_Meteo"] = (Data_clean["Wind_Dir_Meteo"] + 180) % 360
# transformation pour capturer la nature cyclique de la direction du vent
Data_clean["Wind_Dir_Meteo_sin"] = np.sin(2 * np.pi * Data_clean["Wind_Dir_Meteo"] / 360)
Data_clean["Wind_Dir_Meteo_cos"] = np.cos(2 * np.pi * Data_clean["Wind_Dir_Meteo"] / 360)
# suppression des colonnes intermédiaires inutilrs
Data_clean = Data_clean.drop(columns=["Wind_Direction", "Wind_Dir_Meteo"], errors="ignore")

# 2. Calcul de la densité de l'air: ρ= n*M/V - ρ= M*P/(R*T)
# Constantes
R = 8.314  # J/(kg·K) 
M = 0.02896  # kg/mol - Masse molaire de l'air sec
Data_clean["Air_density"]=Data_clean["surface_pressure"]*M/(R*Data_clean["2m_temperature"])

# 3. Création des Lags - features historiques/séquencielles pour le Eolien_MW et Wind_Norm
# Ces lags vont modélser l'autocorrélation de la série temporelle de la production éolienne et du vent
# c'est à dire la dépendance des valeurs actuelles aux valeurs passées. Je me limite à une dépendance temporelle
# de l'ordre de l'heure et de la journée pour éviter l'overfitting.
# La saisonnalité plus long terme sera capturée par les features calendaires 

# 3.1 construction de tags sur Eolien_MW
Data_clean["Y_lag_1h"] = Data_clean["Eolien_MW"].shift(1)   # Lag 1 heure
Data_clean["Y_lag_24h"] = Data_clean["Eolien_MW"].shift(24)  # Lag 24 heures

# 3.2 construction de lags sur la composante de vent normée
Data_clean["Wind_Norm_lag_1h"] = Data_clean["Wind_Norm"].shift(1)
Data_clean["Wind_Norm_lag_24h"] = Data_clean["Wind_Norm"].shift(24)

# 4. création des features calendaires. Elles vont modéliser l'impact du facteur saisonnier.
# C'est à dire l'impact du fait qu'on est en janvier (ça souffle => production) ou en juin (moins de production à priori) 
# 4.1 extraction des composantes calendaires
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

# suppression des features catégorielles brutes Hour, Day, Weekday, Month
Data_clean = Data_clean.drop(columns=["Hour", "Day", "Weekday", "Month"], errors="ignore")

# 5. Intégration des caractéristiques propres aux éoliennes
# 5.1 Création de features basées sur la courbe de puissance d'une éolienne pour capturer le comportement non-linéaire
# on considère une éolienne type avec:
# - une vitesse de cut-in (vitesse minimale pour produire de l'électricité) de 3 m/s
# - une vitesse nominale (vitesse à laquelle l'éolienne produit à pleine capacité) de 12 m/s
# - une vitesse de cut-out (vitesse maximale avant arrêt pour sécurité) de 25 m/s
# P_curve est la puissance théorique d'une éolienne en fonction du vent. Elle est normalisée entre 0 et 1.
# et elle ne dépend que de la vitesse du vent uniquement.
def power_curve(v):
    v_cut_in = 3
    v_nominal = 12
    v_cut_out = 25
    
    if v < v_cut_in:
        return 0.0
    
    elif v < v_nominal:
        # montée physique en (v³ - v_ci³) / (v_r³ - v_ci³)
        return (v**3 - v_cut_in**3) / (v_nominal**3 - v_cut_in**3)
    
    elif v < v_cut_out:
        return 1.0
    
    else:
        return 0.0

Data_clean["P_curve"] = Data_clean["Wind_Norm"].apply(power_curve)

# 5.2 Prise en compte de l'évolution de la taille des parcs éoliens dans le temps
# Les parcs éoliens ont tendance à grossir dans le temps. Donc à météo constante, la production augmente.
# Autrement dit, la production en 2021 ne sera pas la même en 2023. Pour cela, il faut définir:
# - MW_INST: la capacité installée totale des parcs éoliens à chaque date - autrement dit la capacité nominale instantanée.
# C'est donc une constante qu'on peut trouver sur le site de RTE
# - Eolien_MW: puissance réelle produite. Cette puissance est déja connue et extraite des données RTE

#/!\CET ASPECT N'EST PAS PRIS EN COMPTE CAR TROP COMPLEXE /!\ 


# 6. Clean des dernières données
# supprimer la colonne Perimetre/CF et ajout d'une colonne date en index
Data_clean = Data_clean.drop(columns=["Perimetres", "CF"], errors="ignore")
# supprimer les 120 premières lignes avec les valeurs Nan dues aux lags
lags = ["Y_lag_1h", "Y_lag_24h", "Wind_Norm_lag_1h", "Wind_Norm_lag_24h"]
Data_clean = Data_clean.dropna(subset=lags).reset_index(drop=True)

# sauvegarde des données nettoyées
Data_clean.to_csv("Processed_data/data_engineering.csv", index=False)
