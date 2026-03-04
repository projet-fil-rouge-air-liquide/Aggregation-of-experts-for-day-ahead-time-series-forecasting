import pandas as pd
import numpy as np

def build_features(df):
    """Fonction qui construit les features à partir des colonnes
    issues de /Data."""

    rng = np.random.default_rng(42)
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
    is_day = ((hour >= 6) & (hour < 21)).values # True de 6h à 20h sinon false
    is_night = ~is_day # True de 21h à 5h sinon false

    q33 = df["Wind_Norm"].quantile(0.33)
    q66 = df["Wind_Norm"].quantile(0.66)
    mask_low = (df["Wind_Norm"] <= q33).values
    mask_med = ((df["Wind_Norm"] > q33) & (df["Wind_Norm"] <= q66)).values
    mask_high = (df["Wind_Norm"] > q66).values

    def fill_with_noise_every_2_3h(src_col, keep_mask, noise_scale=0.3):
        """
        keep_mask : tableau de booléens issu de build_features()
        - keep_mask = True  : vraie valeur
        - keep_mask = False : avec bruit N(mu, sigma*noise_scale)
          Toutes les 2 ou 3 heures.
          Les heures intermédiaires hors-créneau : interpolation linéaire
          entre les deux points de bruit les plus proches.
        """
        values = df[src_col].values.copy()
        result = np.full(n, np.nan)

        mu = np.nanmean(values[keep_mask]) if keep_mask.any() else np.nanmean(values)
        sigma = np.nanstd(values[keep_mask]) * noise_scale if keep_mask.any() else np.nanstd(values) * noise_scale

        hours_since_noise = 0
        next_interval = int(rng.integers(2, 4))  # intervalle 2 ou 3h tiré aléatoirement

        for i in range(n):
            if keep_mask[i]:
                # Vraie valeur
                result[i] = values[i]
                hours_since_noise = 0
                next_interval = int(rng.integers(2, 4))
            else:
                hours_since_noise += 1
                if hours_since_noise == 1 or hours_since_noise % next_interval == 0:
                    # Bruit toutes les 2 à 3h (choix aléatoire)
                    result[i] = rng.normal(loc=mu, scale=sigma)
                    next_interval = int(rng.integers(2, 4))

        # Interpolation linéaire des NaN entre les points de bruit
        series = pd.Series(result)
        series = series.interpolate(method='linear')
        # Remplir les NaN restants en début/fin
        series = series.bfill().ffill()

        return series.values

    # Features pour le jour : vraies valeurs le jour et bruit toutes les 2-3h (aléatoire) la nuit
    df["P_curve_D"] = fill_with_noise_every_2_3h("P_curve", is_day)
    df["Wind_Norm_D"] = fill_with_noise_every_2_3h("Wind_Norm", is_day)
    df["Air_density_D"] = fill_with_noise_every_2_3h("Air_density", is_day)
    df["Wind_Norm_Cubes_NE_D"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes_NE", is_day)
    df["Wind_Norm_Cubes_NW_D"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes_NW", is_day)
    df["Wind_Norm_Cubes_SE_D"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes_SE", is_day)
    df["Wind_Norm_Cubes_SW_D"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes_SW", is_day)

    # Features pour la nuit : vraies valeurs la nuit et bruit toutes les 2-3h (aléatoire aussi) le jour
    df["P_curve_N"] = fill_with_noise_every_2_3h("P_curve", is_night)
    df["Wind_Norm_N"] = fill_with_noise_every_2_3h("Wind_Norm", is_night)
    df["Air_density_N"] = fill_with_noise_every_2_3h("Air_density", is_night)
    df["Wind_Norm_Cubes_NE_N"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes_NE", is_night)
    df["Wind_Norm_Cubes_NW_N"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes_NW", is_night)
    df["Wind_Norm_Cubes_SE_N"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes_SE", is_night)
    df["Wind_Norm_Cubes_SW_N"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes_SW", is_night)

    # Wind level (bruit plus faible car beaucoup de hors-créneau)
    df["wind_low"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_low, noise_scale=0.1)
    df["wind_med"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_med, noise_scale=0.1)
    df["wind_high"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_high, noise_scale=0.1)

    df["wind_low_D"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_low & is_day, noise_scale=0.1)
    df["wind_med_D"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_med & is_day, noise_scale=0.1)
    df["wind_high_D"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_high & is_day, noise_scale=0.1)
    df["wind_low_N"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_low & is_night, noise_scale=0.1)
    df["wind_med_N"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_med & is_night, noise_scale=0.1)
    df["wind_high_N"] = fill_with_noise_every_2_3h("Wind_Norm_Cubes", mask_high & is_night, noise_scale=0.1)

    return df