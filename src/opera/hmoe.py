import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmmlearn.hmm import GaussianHMM

from src.opera.mixture import Mixture
from src.opera.mixture import HierarchicalHorizonOPERA, RegimeGate


# =========================
# CHARGEMENT DES DONNÉES
# =========================

# Targets et experts
targets = pd.read_csv("data/experts/experts.csv", usecols=[1]).squeeze()
experts = pd.read_csv("data/experts/experts.csv", usecols=[2, 3, 4])

# Features pour le gate de régime
regime_features = pd.read_csv("data/regime_features.csv")

N = len(targets)
horizons = list(range(1, 49))

# =========================
# DÉTECTION DES RÉGIMES (HMM)
# =========================

# Rendements
returns = targets.diff()

df_hmm = pd.DataFrame({
    "ret": returns,
    "vol": returns.rolling(24).std()
}).dropna()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df_hmm.values)

hmm = GaussianHMM(
    n_components=2,
    covariance_type="diag",
    n_iter=300,
    random_state=42
)
hmm.fit(X)

states = hmm.predict(X)

regime_labels = pd.Series(states, index=df_hmm.index)
regime_labels = regime_labels.reindex(targets.index).ffill().values
# =========================
# IDENTIFICATION BULL / BEAR
# =========================

# Moyenne des rendements par état
state_means = [
    returns[regime_labels == i].mean()
    for i in range(2)
]

bull_state = np.argmax(state_means)
bear_state = 1 - bull_state

# 0 = bull, 1 = bear
regime_labels = np.where(regime_labels == bull_state, 0, 1)

# (optionnel) sauvegarde
pd.DataFrame({"regime": regime_labels}).to_csv(
    "data/regimes.csv", index=False
)

# =========================
# MODÈLE HIÉRARCHIQUE OPERA
# =========================

parameters = {
    "eta": 0.05,   # learning rate constant → adaptatif
    "l1": 0.0,
    "l2": 0.01,
}
hmod = HierarchicalHorizonOPERA(
    y=targets.values,
    experts=experts,
    regimes=["bull", "bear"],
    horizons=horizons,
    regime_gate=RegimeGate(n_regimes=2, lr=0.1),
    model="FTRL",
    loss_type="mse",
    parameters=parameters,
)

# =========================
# APPRENTISSAGE ONLINE
# =========================

N_train = N - max(horizons)

for t in tqdm(range(N_train), desc="Opera learning"):
    expert_preds_t = {
        h: experts.iloc[[t]]   # DataFrame 1×n_experts
        for h in horizons
    }
    # shift
    y_true_t = {
        h: targets.iloc[t + h]
        for h in horizons
    }

    hmod.update(
        expert_preds=expert_preds_t,
        y_true=y_true_t,
        regime_features=regime_features.iloc[t].values,
        regime_label=regime_labels[t]
    )

# =========================
# PRÉDICTION 24H
# =========================

new_experts = pd.read_csv(
    "data/experts/pred_24h.csv",
    usecols=[2, 3, 4]
)

expert_preds = {
    h: new_experts
    for h in horizons
}

y_pred_24h = hmod.predict(
    expert_preds=expert_preds,
    regime_features=regime_features.iloc[-1].values
)[24]

# =========================
# POIDS OPERA (DEBUG)
# =========================

print("\nPoids OPERA par régime et horizon :\n")

for r in hmod.regimes:
    for h in [1, 12, 24, 48]:
        w = hmod.opera[r][h].w
        print(f"{r} | h={h} | w={np.round(w, 3)}")

# =========================
# PLOT PRÉDICTION VS VÉRITÉ + EXPERTS
# =========================

df_last24 = pd.read_csv("data/experts/pred_24h.csv")
df_last24["Date_Heure"] = pd.to_datetime(df_last24["Date_Heure"])

# sécurité
for col in ["y_true", "randomforest", "lgbm", "elasticnet"]:
    df_last24[col] = pd.to_numeric(df_last24[col], errors="coerce")

plt.figure(figsize=(15, 7))

# Vérité
plt.plot(
    df_last24["Date_Heure"],
    df_last24["y_true"],
    label="y_true",
    linewidth=2.5,
    color="black"
)

# Experts
plt.plot(
    df_last24["Date_Heure"],
    df_last24["randomforest"],
    label="Expert: RandomForest",
    linestyle="--",
    alpha=0.7
)

plt.plot(
    df_last24["Date_Heure"],
    df_last24["lgbm"],
    label="Expert: LGBM",
    linestyle="--",
    alpha=0.7
)

plt.plot(
    df_last24["Date_Heure"],
    df_last24["elasticnet"],
    label="Expert: ElasticNet",
    linestyle="--",
    alpha=0.7
)

# OPERA
plt.plot(
    df_last24["Date_Heure"],
    y_pred_24h,
    label="OPERA hiérarchique",
    color="red",
    linewidth=2.5
)

plt.title("Prédiction 24h – OPERA hiérarchique vs Experts", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Valeur")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
