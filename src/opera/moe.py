import pandas as pd
from src.opera.mixture import Mixture
import numpy as np
import matplotlib.pyplot as plt

targets = pd.read_csv("data/experts/experts.csv", usecols=[1]) 
targets = targets.squeeze()
experts = pd.read_csv("data/experts/experts.csv", usecols=[2, 3, 4])  

# model = "BOA"
# model = "MLpol"
# model = "MLprod"

set_awake = False

if set_awake:
    awake = np.tile(np.array([1, 0, 1]), (experts.shape[0],)).reshape(experts.shape)
else:
    awake = np.tile(np.array([1, 1, 1]), (experts.shape[0],)).reshape(experts.shape)

N = len(targets)
mod_1 = Mixture(
    y=targets.iloc[0:N],
    experts=experts.iloc[0:N],
    awake=awake[0:N],
    model="FTRL",
    loss_type="mse",
    parameters={
        "fun_reg": lambda w: 0.1 * np.sum(w**2),      # régularisation L2
        "fun_reg_grad": lambda w: 0.2 * w,
        "constraints": [{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        "options": {"maxiter": 50},
    }
)

mod_1.plot_mixture()
mod_1.plot_mixture(plot_type="plot_weight")

new_experts = pd.read_csv("data/experts/pred_24h.csv", usecols=[2, 3, 4])
y_pred_24h = mod_1.predict(new_experts)

print("Prédiction au temps t+1 :", y_pred_24h)

print("\nPoids du modèle (self.w) :")
print(mod_1.w)

print("\nSomme des poids :", mod_1.w.sum())


# plot pred vs GT
df_last24 = pd.read_csv("data/experts/pred_24h.csv")

df_last24["Date_Heure"] = pd.to_datetime(df_last24["Date_Heure"])

for col in ["y_true", "randomforest", "lgbm", "elasticnet"]:
    df_last24[col] = pd.to_numeric(df_last24[col], errors="coerce")

# plot
plt.figure(figsize=(14,6))

plt.plot(df_last24["Date_Heure"], df_last24["y_true"], label="y_true", linewidth=2)
plt.plot(df_last24["Date_Heure"], df_last24["randomforest"], label="randomforest", alpha=0.6)
plt.plot(df_last24["Date_Heure"], df_last24["lgbm"], label="lgbm", alpha=0.6)
plt.plot(df_last24["Date_Heure"], df_last24["elasticnet"], label="elasticnet", alpha=0.6)

plt.plot(df_last24["Date_Heure"], y_pred_24h, label="OPERA", color="red", linewidth=2)

plt.xlabel("Date")
plt.ylabel("Valeur (MW)")
plt.title("OPERA vs Experts vs True (24h Forecast)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
