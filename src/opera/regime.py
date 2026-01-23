import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

experts = pd.read_csv("data/processed_data/data_engineering_belgique.csv")

targets = experts.iloc[:, 1]
dates = experts["Date_Heure"]

#--- features ---
features = pd.DataFrame(index=experts.index)

# Rendement short terme
features["ret_1"] = targets.diff()

# Tendance long terme
features["ret_24"] = targets.diff(24)

# Volatility rolling
features["vol_24"] = targets.diff().rolling(24).std()

# Momentum
features["mom_24"] = targets - targets.shift(24)

# Saisonnality
hour = pd.to_datetime(dates).dt.hour
features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

# Standardisation (feature-wise)
scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features),
    index=features.index,
    columns=features.columns
)

# inject in experts.csv
for col in features_scaled.columns:
    experts[col] = features_scaled[col]

experts.to_csv("data/processed_data/data_engineering_belgique_regime.csv", index=False)
