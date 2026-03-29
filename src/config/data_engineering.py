import pandas as pd
import numpy as np

def build_features(df):
    """Fonction qui construit les features à partir des colonnes
    issues de /Data."""

    n = len(df)

    # Wind Direction Meteo
    df["Wind_Dir_Meteo"] = np.arctan2(df["Wind_Dir_Meteo_sin"], df["Wind_Dir_Meteo_cos"])

    # Wind Norm Cubes par orientation
    df["Wind_Norm_Cubes_NE"] = df["Wind_Norm_Cubes"] * (
        (df["Wind_Dir_Meteo_sin"] > 0) & (df["Wind_Dir_Meteo_cos"] > 0)).astype(float)
    df["Wind_Norm_Cubes_NW"] = df["Wind_Norm_Cubes"] * (
        (df["Wind_Dir_Meteo_sin"] > 0) & (df["Wind_Dir_Meteo_cos"] <= 0)).astype(float)
    df["Wind_Norm_Cubes_SE"] = df["Wind_Norm_Cubes"] * (
        (df["Wind_Dir_Meteo_sin"] <= 0) & (df["Wind_Dir_Meteo_cos"] > 0)).astype(float)
    df["Wind_Norm_Cubes_SW"] = df["Wind_Norm_Cubes"] * (
        (df["Wind_Dir_Meteo_sin"] <= 0) & (df["Wind_Dir_Meteo_cos"] <= 0)).astype(float)

    # Plage jour/ nuit 
    hour = np.round(np.arctan2(df["Hour_sin"], df["Hour_cos"]) * 12 / np.pi) % 24
    # Day = [6h, 20h]
    # Night = [21h, 5h]
    is_day = ((hour >= 6) & (hour < 21)).values
    is_night = ~is_day

    q33 = df["Wind_Norm"].quantile(0.33)
    q66 = df["Wind_Norm"].quantile(0.66)
    mask_low = (df["Wind_Norm"] <= q33).values
    mask_med = ((df["Wind_Norm"] > q33) & (df["Wind_Norm"] <= q66)).values
    mask_high = (df["Wind_Norm"] > q66).values

    def masked_feature(src_col, keep_mask):
        """Vraie valeur où keep_mask est True, 0.0 sinon (vectorisé)."""
        return np.where(keep_mask, df[src_col].values, 0.0)

    # Features pour le jour
    df["P_curve_D"] = masked_feature("P_curve", is_day)
    df["Wind_Norm_D"] = masked_feature("Wind_Norm", is_day)
    df["Air_density_D"] = masked_feature("Air_density", is_day)
    df["Wind_Norm_Cubes_NE_D"] = masked_feature("Wind_Norm_Cubes_NE", is_day)
    df["Wind_Norm_Cubes_NW_D"] = masked_feature("Wind_Norm_Cubes_NW", is_day)
    df["Wind_Norm_Cubes_SE_D"] = masked_feature("Wind_Norm_Cubes_SE", is_day)
    df["Wind_Norm_Cubes_SW_D"] = masked_feature("Wind_Norm_Cubes_SW", is_day)

    # Features pour la nuit
    df["P_curve_N"] = masked_feature("P_curve", is_night)
    df["Wind_Norm_N"] = masked_feature("Wind_Norm", is_night)
    df["Air_density_N"] = masked_feature("Air_density", is_night)
    df["Wind_Norm_Cubes_NE_N"] = masked_feature("Wind_Norm_Cubes_NE", is_night)
    df["Wind_Norm_Cubes_NW_N"] = masked_feature("Wind_Norm_Cubes_NW", is_night)
    df["Wind_Norm_Cubes_SE_N"] = masked_feature("Wind_Norm_Cubes_SE", is_night)
    df["Wind_Norm_Cubes_SW_N"] = masked_feature("Wind_Norm_Cubes_SW", is_night)

    # Wind level
    df["wind_low"] = masked_feature("Wind_Norm_Cubes", mask_low)
    df["wind_med"] = masked_feature("Wind_Norm_Cubes", mask_med)
    df["wind_high"] = masked_feature("Wind_Norm_Cubes", mask_high)

    df["wind_low_D"] = masked_feature("Wind_Norm_Cubes", mask_low & is_day)
    df["wind_med_D"] = masked_feature("Wind_Norm_Cubes", mask_med & is_day)
    df["wind_high_D"] = masked_feature("Wind_Norm_Cubes", mask_high & is_day)
    df["wind_low_N"] = masked_feature("Wind_Norm_Cubes", mask_low & is_night)
    df["wind_med_N"] = masked_feature("Wind_Norm_Cubes", mask_med & is_night)
    df["wind_high_N"] = masked_feature("Wind_Norm_Cubes", mask_high & is_night)

    # Lags jour/ nuit
    df["Y_lag_24h_D"] = masked_feature("Y_lag_24h", is_day)
    df["Wind_Norm_lag_24h_D"] = masked_feature("Wind_Norm_lag_24h", is_day)
    df["Y_lag_24h_N"] = masked_feature("Y_lag_24h", is_night)
    df["Wind_Norm_lag_24h_N"] = masked_feature("Wind_Norm_lag_24h", is_night)

    # --- Saisons & Mois (colonnes ajoutées en bloc pour éviter la fragmentation) ---
    dates = pd.to_datetime(df['Date_Heure'])
    month_day = dates.dt.month * 100 + dates.dt.day

    # 4 saisons
    #is_spring = ((month_day >= 320) & (month_day <= 620)).values
    #is_summer = ((month_day >= 621) & (month_day <= 922)).values
    #is_autumn = ((month_day >= 923) & (month_day <= 1222)).values
    #is_winter = ((month_day >= 1223) | (month_day <= 319)).values
    
    #season_names = ["spring", "summer", "autumn", "winter"]
    #season_masks = [is_spring, is_summer, is_autumn, is_winter]

    # 2 saisons : fusion deux à deux
    is_summer = ((month_day >= 320) & (month_day <= 922)).values
    is_winter = ((month_day >= 923) | (month_day <= 319)).values

    season_names = ["summer", "winter"]
    season_masks = [is_summer, is_winter]


    base_phys = ["P_curve", "Wind_Norm", "Air_density",
                 "Wind_Norm_Cubes_NE", "Wind_Norm_Cubes_NW",
                 "Wind_Norm_Cubes_SE", "Wind_Norm_Cubes_SW"]
    base_lags = ["Y_lag_24h", "Wind_Norm_lag_24h"]

    new_cols = {}

    for sname, smask in zip(season_names, season_masks):
        for col in base_phys + base_lags:
            new_cols[f"{col}_{sname}"] = masked_feature(col, smask)

    # Trimestres (4 experts de 3 mois)
    month_num = dates.dt.month.values
    trimester_names  = ["trim_1", "trim_2", "trim_3", "trim_4"]
    trimester_months = [
        [11, 12, 1],
        [2, 3, 10],
        [4, 5, 9],
        [6, 7, 8],
    ]

    for tname, months in zip(trimester_names, trimester_months):
        tmask = np.isin(month_num, months)
        for col in base_phys + base_lags:
            new_cols[f"{col}_{tname}"] = masked_feature(col, tmask)

    # --- Experts Jour/Nuit bruités (3 niveaux de bruit) ---
    _noisy_base_cols = ["P_curve", "Wind_Norm", "Air_density",
                        "Wind_Norm_Cubes_NE", "Wind_Norm_Cubes_NW",
                        "Wind_Norm_Cubes_SE", "Wind_Norm_Cubes_SW",
                        "wind_low", "wind_med", "wind_high",
                        "Y_lag_24h", "Wind_Norm_lag_24h"]

    # nf = noise faible, nm = noise modéré, nh = noise haut (fort)
    # Le ratio est appliqué à l'écart-type de chaque feature
    noise_levels = {"nf": 0.1, "nm": 0.5, "nh": 0.9}

    rng = np.random.RandomState(42)

    for suffix, ratio in noise_levels.items():
        for col in _noisy_base_cols:
            values = df[col].values
            col_std = values.std()
            # Deux tirages indépendants pour jour et nuit
            noise_d = rng.normal(0, ratio * col_std, size=n)
            noise_n = rng.normal(0, ratio * col_std, size=n)
            # Expert Jour : signal propre le jour, bruité la nuit
            new_cols[f"{col}_D_{suffix}"] = np.where(is_day, values, values + noise_d)
            # Expert Nuit : signal propre la nuit, bruité le jour
            new_cols[f"{col}_N_{suffix}"] = np.where(is_night, values, values + noise_n)

    # Ajout en une seule opération (évite la fragmentation)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df