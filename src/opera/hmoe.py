import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.opera.mixture import Mixture
from src.opera.mixture import HierarchicalHorizonOPERA, RegimeGate


# =========================
# CHARGEMENT DES DONNÉES
# =========================

# Targets et experts
targets = pd.read_csv("data/experts/experts.csv", usecols=[1]).squeeze()
experts = pd.read_csv("data/experts/experts.csv", usecols=[2, 3, 4])


df_last24 = pd.read_csv("data/experts/pred_24h.csv")
df_last24["Date_Heure"] = pd.to_datetime(df_last24["Date_Heure"])

# Features pour le gate de régime
regime_features = pd.read_csv("data/regime_features.csv")

N = len(targets)
horizons = list(range(1, 25)) # toutes les heures jusqu'à +24h

# =========================
# DÉTECTION DES RÉGIMES (HMM)
# =========================

# Rendements
returns = targets.diff()

df_hmm = pd.DataFrame({
    "ret": returns,
    "vol": returns.rolling(24).std()
}).dropna()

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
# DEBUG GATE (PRÉDICTION)
# =========================

x_gate = regime_features.iloc[-1].values
p_regime = hmod.regime_gate.predict(x_gate)

print("\n=== REGIME GATE (prediction time) ===")
for i, r in enumerate(hmod.regimes):
    print(f"P({r}) = {p_regime[i]:.3f}")

print("Régime dominant :", hmod.regimes[np.argmax(p_regime)])


# =========================
# PRÉDICTION 24H
# =========================

new_experts = df_last24.iloc[:, 2:5]


# DEBUG PRÉDICTIONS PAR RÉGIME
print("\n=== PRÉDICTIONS PAR RÉGIME (h=24) ===")

y_hat_regimes = {}

for i, r in enumerate(hmod.regimes):
    X_24 = new_experts.iloc[[0]]  # ou la ligne correspondant à t+24
    y_hat_r = hmod.opera[r][24].predict(X_24).item()
    y_hat_regimes[r] = y_hat_r
    print(f"{r:4s} | ŷ_24 = {float(y_hat_r):.4f}")

# plot bull vs bear
plt.figure(figsize=(15, 6))
for r in hmod.regimes:
    plt.plot(
        df_last24["Date_Heure"],
        [y_hat_regimes[r]] * len(df_last24),
        label=f"OPERA {r}",
        linewidth=2
    )


plt.legend()
plt.grid(alpha=0.3)
plt.title("Prédictions OPERA par régime (24h)")
plt.show()


expert_preds = {
    h: new_experts.iloc[[h-1]]   # h=1 → ligne 0, h=24 → ligne 23
    for h in range(1, 25)
}

y_pred_all = hmod.predict(
    expert_preds=expert_preds,
    regime_features=regime_features.iloc[-1].values
)

print("\n=== COMBINAISON FINALE (h=24) ===")

y_hat_final = 0.0
for i, r in enumerate(hmod.regimes):
    contrib = p_regime[i] * y_hat_regimes[r]
    y_hat_final += contrib
    print(
        f"{r:4s} | p={p_regime[i]:.3f} "
        f"| contrib={float(contrib):.4f}"
    )

print(f"\nŷ_final_24h = {float(y_hat_final):.4f}")

# =========================
# POIDS OPERA
# =========================

print("\n=== POIDS OPERA (internes, par régime et horizon) ===")

for r in hmod.regimes:
    print(f"\nRégime: {r}")
    for h in [1, 8, 16, 24]:
        w = hmod.opera[r][h].w
        print(f"  h={h:2d} | w={np.round(w, 3)}")


print("\n=== POIDS EFFECTIFS PAR EXPERT (h=24) ===")

w_eff = np.zeros(len(experts.columns))

for i, r in enumerate(hmod.regimes):
    w_r = hmod.opera[r][24].w
    w_eff += p_regime[i] * w_r

for name, w in zip(experts.columns, w_eff):
    print(f"{name:15s} | w_eff = {w:.3f}")


# =========================
# PLOT PRÉDICTION VS VÉRITÉ + EXPERTS
# =========================

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
y_pred_curve = [float(y_pred_all[h]) for h in range(1, 25)]
plt.plot(
    df_last24["Date_Heure"],
    y_pred_curve,
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
