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
STATIONAR = ["Y_lag_24h"]
# Hybride à faire

# expert global
GLOBAL = [f for f in cyclique + physique + lag if f not in ["Wind_Norm_Cubes"]]

# Wind direction x4
WIND_ORIENTATION = ["Air_density"] #+ cyclique
WIND_ORIENTATION_NE = WIND_ORIENTATION + ["Wind_Norm_Cubes_NE"]
WIND_ORIENTATION_NW = WIND_ORIENTATION + ["Wind_Norm_Cubes_NW"]
WIND_ORIENTATION_SE = WIND_ORIENTATION + ["Wind_Norm_Cubes_SE"]
WIND_ORIENTATION_SW = WIND_ORIENTATION + ["Wind_Norm_Cubes_SW"]

#Experts Jour/Nuit -> important : créer (2*d) variables et non (2*d + d) variables. On risque d'introduire beaucoup de colinéarité sinon.
DAY = day #+ cyclique 
NIGHT = night #+ cyclique 

# - Experts Forces de vent(faible/fort/moyen)
WIND_LEVEL_LOW =  ["wind_low"] + ["Air_density"]#, "Y_lag_24h"] #+ cyclique 
WIND_LEVEL_MED = ["wind_med"] + ["Air_density"]#, "Y_lag_24h"]#+ cyclique 
WIND_LEVEL_HIGH =  ["wind_high"] + ["Air_density"]#, "Y_lag_24h"]#+ cyclique 

SYNOPTIQUE = ["2m_temperature", "surface_pressure", "Air_density"]

# création d'un dictionnaire de features
features_groupe={"Stationnar":STATIONAR,
                 "Wind_orientation_NE":WIND_ORIENTATION_NE,
                 "Wind_orientation_SE":WIND_ORIENTATION_SE,
                 "Wind_orientation_NW":WIND_ORIENTATION_NW,
                 "Wind_orientation_SW":WIND_ORIENTATION_SW,
                 "Night":NIGHT,
                 "Day":DAY,
                 "Wind_Low": WIND_LEVEL_LOW,
                 "Wind_Med": WIND_LEVEL_MED,
                 "Wind_High": WIND_LEVEL_HIGH,
                 "Global": GLOBAL,
                 "Synoptique": SYNOPTIQUE
                 }