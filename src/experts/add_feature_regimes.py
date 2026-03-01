"""
Add regimes features in expert csv:

TREND (bear/bull):
    - trend_strength
    - mom_48
    - mom_24
    - vol_24
WIND (low/high):
    - Wind_Norm
    - Wind_mean_3h
    - Wind_Norm_lag_1h
    - Wind_Norm_lag_24h
"""
import numpy as np
import pandas as pd

# Load Data
experts = pd.read_csv("data/experts/experts.csv")
meteo = pd.read_csv("data/processed_data/data_engineering_belgique.csv")

experts["Date_Heure"] = pd.to_datetime(experts["Date_Heure"])
meteo["Date_Heure"] = pd.to_datetime(meteo["Date_Heure"])

dates = experts["Date_Heure"]
targets = experts["y_true"]


features = pd.DataFrame(index=experts.index)

rets = targets.diff()

# Momentum
features["mom_24"] = (targets - targets.shift(24)).shift(1)
features["mom_48"] = (targets - targets.shift(48)).shift(1)

# Past Volatility
features["vol_12"] = rets.rolling(12).std().shift(1)
features["vol_24"] = rets.rolling(24).std().shift(1)

# Trend Strenght
features["trend_strength"] = (
    rets.rolling(24).mean() /
    (rets.rolling(24).std() + 1e-8)
).shift(1)

features["Date_Heure"] = dates.values

# Merge weather
feat_wind = [
    "Wind_Norm",
    "Wind_mean_3h",
    "Wind_Norm_lag_1h",
    "Wind_Norm_lag_24h",
]

features = features.merge(
    meteo[["Date_Heure"] + feat_wind],
    on="Date_Heure",
    how="left"
)

# --- Test add experts ---
features["randomforest_plus"] = experts["randomforest"] + 300

features["lgbm_minus"] = experts["lgbm"] - 200

##########################

# Export
final_data = pd.concat(
    [
        experts[
            ["Date_Heure", "y_true", "randomforest", "lgbm", "elasticnet"]
        ],
        features.drop(columns="Date_Heure"),
    ],
    axis=1
)

final_data = final_data.dropna().reset_index(drop=True)

final_data.to_csv(
    "data/experts/experts_features.csv",
    index=False
)