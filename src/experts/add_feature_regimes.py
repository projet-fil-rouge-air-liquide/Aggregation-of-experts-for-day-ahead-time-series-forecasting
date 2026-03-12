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
experts = pd.read_csv("data/experts/predictions_experts_globaux_10000.csv")
meteo = pd.read_csv("data/processed_data/data_engineering_belgique.csv")

experts["Date_Heure"] = pd.to_datetime(experts["Date_Heure"])
meteo["Date_Heure"] = pd.to_datetime(meteo["Date_Heure"])

dates = experts["Date_Heure"]
targets = experts["y_true"]

# Regime features
features = pd.DataFrame(index=experts.index)

rets = targets.diff()

# Momentum
features["mom_24"] = (targets - targets.shift(24)).shift(1)
features["mom_48"] = (targets - targets.shift(48)).shift(1)

# Past Volatility
features["vol_12"] = rets.rolling(12).std().shift(1)
features["vol_24"] = rets.rolling(24).std().shift(1)

# Trend Strength
features["trend_strength"] = (
    rets.rolling(24).mean() /
    (rets.rolling(24).std() + 1e-8)
).shift(1)

features["Date_Heure"] = dates.values

# Merge weather features
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

# Combine with experts
expert_cols = [c for c in experts.columns if c not in ["Date_Heure", "y_true"]]

final_data = pd.concat(
    [
        experts[["Date_Heure", "y_true"] + expert_cols],
        features.drop(columns="Date_Heure"),
    ],
    axis=1
)

final_data = final_data.reset_index(drop=True)

final_data.to_csv(
    "data/experts/predictions_experts_spe_feat_10000.csv",
    index=False
)