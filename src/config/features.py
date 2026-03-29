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
# Exemple : Cela force l'expert Day à ne plus s'appuyer sur 
# le lag pour prédire la nuit :
lag_day = ["Y_lag_24h_D", "Wind_Norm_lag_24h_D"]
lag_night = ["Y_lag_24h_N", "Wind_Norm_lag_24h_N"]
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
#WIND_ORIENTATION_NE = GLOBAL + ["Wind_Norm_Cubes_NE"]
#WIND_ORIENTATION_NW = GLOBAL + ["Wind_Norm_Cubes_NW"]
#WIND_ORIENTATION_SE = GLOBAL + ["Wind_Norm_Cubes_SE"]
#WIND_ORIENTATION_SW = GLOBAL + ["Wind_Norm_Cubes_SW"]

# Wind direction : exclusif donc sans GLOBAL
WIND_ORIENTATION_NE = cyclique + lag + ["Wind_Norm_Cubes_NE"]
WIND_ORIENTATION_NW = cyclique + lag + ["Wind_Norm_Cubes_NW"]
WIND_ORIENTATION_SE = cyclique + lag + ["Wind_Norm_Cubes_SE"]
WIND_ORIENTATION_SW = cyclique + lag + ["Wind_Norm_Cubes_SW"]


# Experts Jour/Nuit : GLOBAL + features jour/nuit spécifiques
#DAY = GLOBAL + day

#Ici l'expert Day ne voit que P_curve_D, Wind_Norm_D, etc
# donc le signal réel le jour et 0 la nuit
DAY = cyclique + lag_day + day
#NIGHT = GLOBAL + night

#De même pour l'expert nuit
NIGHT = cyclique + lag_night + night

# ========== Experts Jour/Nuit bruités ==========
# Spectre de spécialisation :
#   Global     = aucun bruit  (voit tout proprement)
#   Noise_Faible  = 10% σ de bruit sur la période "off"
#   Noise_Modere  = 50% σ de bruit sur la période "off"
#   Noise_Fort    = 90% σ de bruit sur la période "off"
#   Day / Night   = masqué à 0  (spécialisation maximale)

_noisy_base = ["P_curve", "Wind_Norm", "Air_density",
               "Wind_Norm_Cubes_NE", "Wind_Norm_Cubes_NW",
               "Wind_Norm_Cubes_SE", "Wind_Norm_Cubes_SW",
               "wind_low", "wind_med", "wind_high",
               "Y_lag_24h", "Wind_Norm_lag_24h"]

DAY_NOISE_FAIBLE = cyclique + [f"{c}_D_nf" for c in _noisy_base]
DAY_NOISE_MODERE = cyclique + [f"{c}_D_nm" for c in _noisy_base]
DAY_NOISE_FORT   = cyclique + [f"{c}_D_nh" for c in _noisy_base]

NIGHT_NOISE_FAIBLE = cyclique + [f"{c}_N_nf" for c in _noisy_base]
NIGHT_NOISE_MODERE = cyclique + [f"{c}_N_nm" for c in _noisy_base]
NIGHT_NOISE_FORT   = cyclique + [f"{c}_N_nh" for c in _noisy_base]

# Experts Forces de vent : GLOBAL + indicateurs de régime
#WIND_LEVEL_LOW = GLOBAL + ["wind_low"]
#WIND_LEVEL_MED = GLOBAL + ["wind_med"]
#WIND_LEVEL_HIGH = GLOBAL + ["wind_high"]

# Wind Level : exclusif donc sans GLOBAL
WIND_LEVEL_LOW = cyclique + lag + ["wind_low"]
WIND_LEVEL_MED = cyclique + lag + ["wind_med"]
WIND_LEVEL_HIGH = cyclique + lag + ["wind_high"]

# Wind Level : exclusif donc sans GLOBAL et test sans lag
#WIND_LEVEL_LOW = cyclique + ["wind_low"]
#WIND_LEVEL_MED = cyclique + ["wind_med"]
#WIND_LEVEL_HIGH = cyclique + ["wind_high"]

# Expert synoptique : GLOBAL + variables météo
#SYNOPTIQUE = GLOBAL + ["2m_temperature", "surface_pressure"]

# Synoptique : exclusif donc sans GLOBAL
SYNOPTIQUE = cyclique + lag + ["2m_temperature", "surface_pressure"]

# Expert avec toutes les orientations
GLOBAL_ALL_ORIENT = GLOBAL + wind_orientation

# Expert avec tous les régimes de vent
GLOBAL_ALL_WIND = GLOBAL + wind_variation


# --- Experts par saison ---
_base_phys = ["P_curve", "Wind_Norm", "Air_density",
              "Wind_Norm_Cubes_NE", "Wind_Norm_Cubes_NW",
              "Wind_Norm_Cubes_SE", "Wind_Norm_Cubes_SW"]
# Pour les 4 saisons
#spring = [f"{c}_spring" for c in _base_phys]
#lag_spring = ["Y_lag_24h_spring", "Wind_Norm_lag_24h_spring"]
#SPRING = cyclique + lag_spring + spring

summer = [f"{c}_summer" for c in _base_phys]
lag_summer = ["Y_lag_24h_summer", "Wind_Norm_lag_24h_summer"]
SUMMER = cyclique + lag_summer + summer

# Pour les 4 saisons
#autumn = [f"{c}_autumn" for c in _base_phys]
#lag_autumn = ["Y_lag_24h_autumn", "Wind_Norm_lag_24h_autumn"]
#AUTUMN = cyclique + lag_autumn + autumn

winter = [f"{c}_winter" for c in _base_phys]
lag_winter = ["Y_lag_24h_winter", "Wind_Norm_lag_24h_winter"]
WINTER = cyclique + lag_winter + winter

# --- Experts par trimestre (4 × 3 mois, regroupement configurable) ---
# Trim 1 : Nov, Déc, Jan  (très venteux)
# Trim 2 : Fév, Mar, Oct  (venteux)
# Trim 3 : Avr, Mai, Sep  (transition)
# Trim 4 : Jun, Jul, Aoû  (calmes)
_trimester_names  = ["trim_1", "trim_2", "trim_3", "trim_4"]
_trimester_labels = ["Trim_Venteux_Fort", "Trim_Venteux", "Trim_Transition", "Trim_Calme"]

TRIMESTER_EXPERTS = {}
for _tname, _tlabel in zip(_trimester_names, _trimester_labels):
    _phys = [f"{c}_{_tname}" for c in _base_phys]
    _lag  = [f"Y_lag_24h_{_tname}", f"Wind_Norm_lag_24h_{_tname}"]
    TRIMESTER_EXPERTS[_tlabel] = cyclique + _lag + _phys

# création d'un dictionnaire de features
features_groupe = {
    "Global": GLOBAL,
    "Global_all_orient": GLOBAL_ALL_ORIENT,
    "Global_all_wind": GLOBAL_ALL_WIND,
    "Wind_orientation_NE": WIND_ORIENTATION_NE,
    "Wind_orientation_SE": WIND_ORIENTATION_SE,
    "Wind_orientation_NW": WIND_ORIENTATION_NW,
    "Wind_orientation_SW": WIND_ORIENTATION_SW,
    #Experts nuit et jour et ceux bruités
    "Night": NIGHT,
    "Day": DAY,
    "Day_Noise_Faible": DAY_NOISE_FAIBLE,
    "Day_Noise_Modere": DAY_NOISE_MODERE,
    "Day_Noise_Fort": DAY_NOISE_FORT,
    "Night_Noise_Faible": NIGHT_NOISE_FAIBLE,
    "Night_Noise_Modere": NIGHT_NOISE_MODERE,
    "Night_Noise_Fort": NIGHT_NOISE_FORT,
    "Wind_Low": WIND_LEVEL_LOW,
    "Wind_Med": WIND_LEVEL_MED,
    "Wind_High": WIND_LEVEL_HIGH,
    "Synoptique": SYNOPTIQUE,
    "Stationnar": STATIONAR,
    # Seasons
    #"Spring": SPRING,
    "Summer": SUMMER,
    #"Autumn": AUTUMN,
    "Winter": WINTER,
    # Experts selon les mois
    **TRIMESTER_EXPERTS,
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