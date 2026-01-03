import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# Chargement de la série cible
# =========================

targets = pd.read_csv(
    "data/experts/experts.csv",
    usecols=[1]
).squeeze()

dates = pd.read_csv(
    "data/experts/experts.csv",
    usecols=[0]
)["Date_Heure"]

# =========================
# Construction des features
# =========================

df = pd.DataFrame(index=targets.index)

# Rendement court terme
df["ret_1"] = targets.diff()

# Tendance long terme
df["ret_24"] = targets.diff(24)

# Volatilité rolling
df["vol_24"] = targets.diff().rolling(24).std()

# Momentum
df["mom_24"] = targets - targets.shift(24)

# (optionnel) saisonnalité
df["hour"] = pd.to_datetime(dates).dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# =========================
# Nettoyage
# =========================

df = df.dropna()

# =========================
# Standardisation (TRÈS IMPORTANT)
# =========================

scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    index=df.index,
    columns=df.columns
)

# =========================
# Sauvegarde
# =========================

df_scaled.to_csv("data/regime_features.csv", index=False)
