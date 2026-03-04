cyclique = ["Hour_sin","Hour_cos","Weekday_sin","Weekday_cos","Wind_Dir_Meteo_cos",
            "Wind_Dir_Meteo_sin","Month_sin","Month_cos"]
wind_orientation = ["Wind_Norm_Cubes_NE","Wind_Norm_Cubes_NW","Wind_Norm_Cubes_SE","Wind_Norm_Cubes_SW"]
physique = ["P_curve","Wind_Norm","Wind_Norm_Cubes","Air_density","Wind_Dir_Meteo"]
day = ["P_curve_D","Wind_Norm_D","Wind_Norm_Cubes_NE_D","Wind_Norm_Cubes_NW_D",
       "Wind_Norm_Cubes_SE_D","Wind_Norm_Cubes_SW_D","Air_density_D",
       "wind_low_D","wind_med_D","wind_high_D"]
night = ["P_curve_N","Wind_Norm_N","Wind_Norm_Cubes_NE_N","Wind_Norm_Cubes_NW_N",
         "Wind_Norm_Cubes_SE_N","Wind_Norm_Cubes_SW_N","Air_density_N",
         "wind_low_N","wind_med_N","wind_high_N"]
lag = ["Y_lag_24h","Wind_Norm_lag_24h"]
wind_variation = ["wind_low","wind_med","wind_high"]
inutiles =["speed_longitudinale_100m","speed_latitudinale_100m","2m_temperature",
           "surface_pressure","mean_sea_level_pressure","sea_surface_temperature","Wind_Norm_10m",
           "Wind_Norm_Cubes_10m","Wind_mean_3h_10m"]

# expert stationnaire/hybride
STATIONAR = ["Y_lag_24h","Wind_Norm_lag_24h"] + cyclique

# expert global et base des features pour les experts spécialisés
GLOBAL = [f for f in cyclique + physique + lag if f not in ["Wind_Norm_Cubes"]]

# ********************** Test **********************
# Proposition de création d'experts spécialisés :Un expert spécialisés =
# features globales + features spécifiques
# Le principe : chaque expert voit tout ce que voit le Global, plus une
# information supplémentaire spécifique.
# Donc Lasso identifiera que l'expert Day est meilleur le jour,
# l'expert Night meilleur la nuit (...) et les combiner intelligemment.

# Wind direction : GLOBAL + le cube directionnel spécifique
WIND_ORIENTATION_NE = GLOBAL + ["Wind_Norm_Cubes_NE"]
WIND_ORIENTATION_NW = GLOBAL + ["Wind_Norm_Cubes_NW"]
WIND_ORIENTATION_SE = GLOBAL + ["Wind_Norm_Cubes_SE"]
WIND_ORIENTATION_SW = GLOBAL + ["Wind_Norm_Cubes_SW"]

# Experts Jour/Nuit : GLOBAL + features jour/nuit spécifiques
DAY = GLOBAL + day
NIGHT = GLOBAL + night

# Experts Forces de vent : GLOBAL + indicateurs de régime
WIND_LEVEL_LOW = GLOBAL + ["wind_low"]
WIND_LEVEL_MED = GLOBAL + ["wind_med"]
WIND_LEVEL_HIGH = GLOBAL + ["wind_high"]

# Expert synoptique : GLOBAL + variables météo
SYNOPTIQUE = GLOBAL + ["2m_temperature", "surface_pressure"]

# Expert avec toutes les orientations
GLOBAL_ALL_ORIENT = GLOBAL + wind_orientation

# Expert avec tous les régimes de vent
GLOBAL_ALL_WIND = GLOBAL + wind_variation

# création d'un dictionnaire de features
features_groupe = {
    "Global": GLOBAL,
    "Global_all_orient": GLOBAL_ALL_ORIENT,
    "Global_all_wind": GLOBAL_ALL_WIND,
    "Wind_orientation_NE": WIND_ORIENTATION_NE,
    "Wind_orientation_SE": WIND_ORIENTATION_SE,
    "Wind_orientation_NW": WIND_ORIENTATION_NW,
    "Wind_orientation_SW": WIND_ORIENTATION_SW,
    "Night": NIGHT,
    "Day": DAY,
    "Wind_Low": WIND_LEVEL_LOW,
    "Wind_Med": WIND_LEVEL_MED,
    "Wind_High": WIND_LEVEL_HIGH,
    "Synoptique": SYNOPTIQUE,
    "Stationnar": STATIONAR,
}




# ********************** AVANT **********************
# Wind direction x4
#WIND_ORIENTATION = ["Air_density"] + cyclique
#WIND_ORIENTATION_NE = WIND_ORIENTATION + ["Wind_Norm_Cubes_NE"]
#WIND_ORIENTATION_NW = WIND_ORIENTATION + ["Wind_Norm_Cubes_NW"]
#WIND_ORIENTATION_SE = WIND_ORIENTATION + ["Wind_Norm_Cubes_SE"]
#WIND_ORIENTATION_SW = WIND_ORIENTATION + ["Wind_Norm_Cubes_SW"]

# Experts Jour/Nuit
#DAY = day + cyclique
#NIGHT = night + cyclique

# Experts Forces de vent (faible/fort/moyen)
#WIND_LEVEL_LOW = ["wind_low", "Air_density", "Y_lag_24h"] + cyclique
#WIND_LEVEL_MED = ["wind_med", "Air_density", "Y_lag_24h"] + cyclique
#WIND_LEVEL_HIGH = ["wind_high", "Air_density", "Y_lag_24h"] + cyclique

#SYNOPTIQUE = ["2m_temperature", "surface_pressure", "Air_density"] + cyclique

# création d'un dictionnaire de features
#features_groupe={"Stationnar":STATIONAR,
#                 "Wind_orientation_NE":WIND_ORIENTATION_NE,
#                 "Wind_orientation_SE":WIND_ORIENTATION_SE,
#                 "Wind_orientation_NW":WIND_ORIENTATION_NW,
#                 "Wind_orientation_SW":WIND_ORIENTATION_SW,
#                 "Night":NIGHT,
#                 "Day":DAY,
#                 "Wind_Low": WIND_LEVEL_LOW,
#                 "Wind_Med": WIND_LEVEL_MED,
#                 "Wind_High": WIND_LEVEL_HIGH,
#                 "Global": GLOBAL,
#                 "Synoptique": SYNOPTIQUE
#                 }