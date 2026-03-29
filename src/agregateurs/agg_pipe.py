import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor

from utils.fonction import fit_predict_eval, calculate_nmae, predict_eval

from src.config.data_train_valid_test import X_train,X_valid,X_test,y_train,y_valid,y_test
from src.config.data_train_valid_test_to_24 import X_train_24,X_valid_24,X_test_24,y_train_24,y_valid_24,y_test_24

from src.config.features import features_groupe
from src.experts import expert_LGBM,expert_Ridge,expert_RandomForest
#from src.agregateurs.agg_lin import AGG_LIN

#Permet d'avoir la date et l'heure dans les fichiers créés : plot + csv d'export
from datetime import datetime
_ts = datetime.now().strftime("%Y-%m-%d_%Hh%M")

import sys

_log_path = f"Data/output_{_ts}.log"
_log_file = open(_log_path, "w")

class _Tee:
    """Permet d'écire simultanément dans le 
    terminal et dans un fichier (la sortie)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = _Tee(sys.__stdout__, _log_file)

# classes d'expert
expert_classe = [expert_Ridge.RidgeExpert,
                 expert_RandomForest.RandomForestExpert,
                 expert_LGBM.LGBMExpert
                 ]

experts=[]
for name,features in features_groupe.items():
    for exp in expert_classe:
        experts.append(exp(features=features,features_name=name))

results=[]
experts_preds_val =[]
experts_preds_test = []
ref_capacity = max(y_train.max(), y_valid.max(), y_test.max())

for exp in experts:
    exp.fit(X_train,y_train)

    y_pred, wape_value, mae, mse, rmse, mape, nmae_value = predict_eval(
        exp, 
        X_test, 
        y_test, 
        capacity=ref_capacity
    ) 

# construction des variables pour l'agrégation
    # variables validation
    y_pred_val = exp.predict(X_valid) 
    experts_preds_val.append(y_pred_val)
    # variables test
    experts_preds_test.append(y_pred.flatten())

    results.append({
        "Exp_name": f"{exp.name}_{exp.features_name}",
        "nmae_%": nmae_value,
        "wape": wape_value,
        "mae": mae,
        "rmse": rmse,
        "mape_%": mape,
        "mse": mse,
    })
    
results = pd.DataFrame(results)
results["model"] = results["Exp_name"].str.split("_", n=1).str[0]
results.sort_values(by=["model", "nmae_%"], ascending=True, inplace=True)
print(results.drop(columns="model").to_string(index=False))
results.drop(columns="model").to_csv(f"Data/results_experts_{_ts}.csv", index=False, sep=";")
print(f"CSV sauvegardé : results_experts_{_ts}.csv ({results.shape})")

# conditionnemnet des variables d'agrégation avec np.column_stack
y_valid_flat = y_valid.values.flatten()
X_val_agg = np.column_stack(experts_preds_val)
X_test_agg = np.column_stack(experts_preds_test)

# ---------------- création de l'agrégateur linéaire avec cross validation - RIDGE ---------------- #
tsv = TimeSeriesSplit(n_splits=5)
alpha_test = np.logspace(2,10,20)
agg_ridge_cv = RidgeCV(alphas=alpha_test,
                       cv=tsv,
                       scoring='neg_mean_absolute_error',
                       fit_intercept=False)
agg_ridge_cv.fit(X_val_agg,y_valid_flat)

alpha_best = agg_ridge_cv.alpha_
print('Ridge best alpha: ',alpha_best)

y_pred_agg_test, wape_agg, mae_agg, mse_agg, rmse_agg, mape_agg, nmae_agg = predict_eval(
    agg_ridge_cv,
    X_test_agg,
    y_test,
    capacity=ref_capacity
)
# récupération des coefficients
coefficient = agg_ridge_cv.coef_        
intercept = agg_ridge_cv.intercept_

df_coef = pd.DataFrame({
    'coefficient':coefficient,
    'intercept':intercept,
    "expert": [f"{exp.name}_{exp.features_name}" for exp in experts]
})
#print(df_coef)
print(f"nMAE de l'agrégateur Ridge cv : {nmae_agg:.2f}% | RMSE : {rmse_agg:.2f} | MAPE : {mape_agg:.2f}%")

# ---------------- création de l'agrégateur linéaire avec cross validation - LASSO ---------------- #
tsv_l = TimeSeriesSplit(5)
alpha_test = np.logspace(-4,4,20)
agg_lasso_cv = LassoCV(
    alphas=alpha_test,
    cv=tsv,
    random_state=0,
    max_iter=10000,
    fit_intercept=False,
    selection='random',
    n_jobs=-1,
)

agg_lasso_cv.fit(X_val_agg,y_valid_flat)

#alpha_best = agg_lasso_cv.alpha_
#print('best alpha: ',alpha_best)

y_pred_agg_test, wape_agg, mae_agg_la, mse_agg, rmse_agg_la, mape_agg_la, nmae_agg_la = predict_eval(
    agg_lasso_cv,
    X_test_agg,
    y_test,
    capacity=ref_capacity
)
coef_lasso = agg_lasso_cv.coef_
experts_conserves = np.sum(coef_lasso != 0)
print(f"Lasso meilleur alpha : {agg_lasso_cv.alpha_}")
print(f"Nombre d'experts conservés par le Lasso : {experts_conserves} / {len(coef_lasso)}")
print(f"nMAE de l'agrégateur Lasso : {nmae_agg_la:.2f}% | RMSE : {rmse_agg_la:.2f} | MAPE : {mape_agg_la:.2f}%")

# Liste des experts conservés avec leurs coefficients
df_coef_lasso = pd.DataFrame({
    'expert': [f"{exp.name}_{exp.features_name}" for exp in experts],
    'coefficient': coef_lasso
})
df_coef_lasso = df_coef_lasso[df_coef_lasso['coefficient'] != 0]
df_coef_lasso["model"] = df_coef_lasso["expert"].str.split("_", n=1).str[0]
df_coef_lasso.sort_values(by=["model", "coefficient"], ascending=[True, False], inplace=True)
print("\nExperts conservés par le Lasso :")
print(df_coef_lasso.drop(columns="model").to_string(index=False))

# ---------------- Export CSV des prédictions de tous les experts ----------------
expert_names = [f"{exp.name}_{exp.features_name}" for exp in experts]

df_preds_lasso = pd.DataFrame({
    "Date_Heure": X_test["Date_Heure"].values,
    "y_true": y_test.values.flatten(),
})

for name, preds in zip(expert_names, experts_preds_test):
    df_preds_lasso[name] = preds

# Ne garder que les experts globaux
#global_cols = ["Date_Heure", "y_true"] + [c for c in df_preds_lasso.columns if c.endswith("_Global")]
#df_preds_lasso = df_preds_lasso[global_cols]

df_preds_lasso.to_csv(f"Data/predictions_experts_{_ts}.csv", index=False, sep=";")
print(f"\nCSV sauvegardé : predictions_experts_{_ts}.csv ({df_preds_lasso.shape})")

# ---------------- création de l'agrégateur non linéaire  ----------------

agg_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=10,
    random_state=10,
    n_jobs=-1
)
agg_rf.fit(X_val_agg,y_valid_flat)

y_pred_agg_test, wape_agg_rf, mae_agg_rf, mse_agg_rf, rmse_agg_rf, mape_agg_rf, nmae_agg_rf = predict_eval(
    agg_rf,
    X_test_agg,
    y_test,
    capacity=ref_capacity
)
print(f"nMAE de l'agrégateur RF : {nmae_agg_rf:.2f}% | RMSE : {rmse_agg_rf:.2f} | MAPE : {mape_agg_rf:.2f}%")


import matplotlib.pyplot as plt
import numpy as np

def plot_day_ahead_lasso(start_day, n_days=5, df_csv_path="Data/predictions_experts.csv"):
    """Visualise les prédictions des experts Lasso vs y_true sur n_days jours."""
    df = pd.read_csv(df_csv_path, parse_dates=["Date_Heure"], sep=";")    
    start = start_day * 24
    end = start + n_days * 24
    period_df = df.iloc[start:end]
    
    plt.figure(figsize=(18, 6))
    
    hours = range(len(period_df))
    
    # Bandes jour/nuit
    for d in range(n_days):
        base = d * 24
        # Nuit début : 0h-5h (heures 0 à 5)
        plt.axvspan(base, base + 6, color='midnightblue', alpha=0.08)
        # Jour : 6h-20h (heures 6 à 20)
        plt.axvspan(base + 6, base + 21, color='gold', alpha=0.08)
        # Nuit fin : 21h-23h (heures 21 à 23)
        plt.axvspan(base + 21, base + 24, color='midnightblue', alpha=0.08)
    
    plt.plot(hours, period_df["y_true"].values, color='black', linewidth=2, label="y_true", zorder=10)
    
    for col in ["LGBM_Night", "LGBM_Day"]:
        if col in period_df.columns:
            plt.plot(hours, period_df[col].values, '--', linewidth=1.5, alpha=0.7, label=col)
    
    # Lignes verticales pour séparer les jours
    for d in range(1, n_days):
        plt.axvline(x=d * 24, color='grey', linestyle=':', alpha=0.5)
    
    # Labels sur l'axe x : une date par jour
    tick_pos = [d * 24 for d in range(n_days)]
    tick_labels = [period_df["Date_Heure"].iloc[d * 24].strftime("%Y-%m-%d") for d in range(n_days)]
    plt.xticks(tick_pos, tick_labels, rotation=45)
    
    date_start = period_df["Date_Heure"].iloc[0].strftime("%Y-%m-%d")
    date_end = period_df["Date_Heure"].iloc[-1].strftime("%Y-%m-%d")
    plt.title(f"Profil Day-Ahead — {date_start} à {date_end}")
    plt.xlabel("Date")
    plt.ylabel("Puissance (MW)")
    
    # Légende jour/nuit
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='y_true'),
        plt.Line2D([0], [0], color='tab:blue', linestyle='--', label='LGBM_Night'),
        plt.Line2D([0], [0], color='tab:orange', linestyle='--', label='LGBM_Day'),
        Patch(facecolor='gold', alpha=0.2, label='Jour (6h–20h)'),
        Patch(facecolor='midnightblue', alpha=0.2, label='Nuit (21h–5h)'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Data/plot_day_ahead_{date_start}_to_{date_end}_{_ts}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot sauvegardé : Data/plot_day_ahead_{date_start}_to_{date_end}_{_ts}.png")

# ---------------- comparaison experts jour/nuit bruités ----------------
def plot_day_night_noise_comparison(start_day, n_days=7, df_csv_path=None, model="LGBM"):
    """Compare les experts jour/nuit bruités vs masqués vs Global sur n_days."""
    df = pd.read_csv(df_csv_path, parse_dates=["Date_Heure"], sep=";")
    start = start_day * 24
    end = start + n_days * 24
    period_df = df.iloc[start:end]

    hours = range(len(period_df))

    fig, (ax_day, ax_night) = plt.subplots(2, 1, figsize=(22, 12), sharex=True)

    # Bandes jour et nuit
    for ax in (ax_day, ax_night):
        for d in range(n_days):
            base = d * 24
            ax.axvspan(base, base + 6, color='midnightblue', alpha=0.08)
            ax.axvspan(base + 6, base + 21, color='gold', alpha=0.08)
            ax.axvspan(base + 21, base + 24, color='midnightblue', alpha=0.08)
        for d in range(1, n_days):
            ax.axvline(x=d * 24, color='grey', linestyle=':', alpha=0.5)

    # Jour
    ax_day.plot(hours, period_df["y_true"].values,
                color='black', linewidth=2, label="y_true", zorder=10)
    day_experts = [
        (f"{model}_Global",           "gray",    "-",  1.2, "Global (ref)"),
        (f"{model}_Day_Noise_Faible", "#FF00D9", "--", 1.2, "Day bruit faible (10%)"),
        (f"{model}_Day_Noise_Modere", "#88FF00", "--", 1.2, "Day bruit modéré (50%)"),
        (f"{model}_Day_Noise_Fort",   "#FF4500", "--", 1.2, "Day bruit fort (90%)"),
        (f"{model}_Day",              "#FFEE03", "-",  1.5, "Day masqué (actuel)"),
    ]
    for col, color, ls, lw, label in day_experts:
        if col in period_df.columns:
            ax_day.plot(hours, period_df[col].values, ls, color=color,
                        linewidth=lw, alpha=0.8, label=label)
    ax_day.set_title(f"Experts JOUR — Comparaison niveaux de bruit ({model})")
    ax_day.set_ylabel("Puissance (MW)")
    ax_day.grid(True, alpha=0.3)

    # Nuit
    ax_night.plot(hours, period_df["y_true"].values,
                  color='black', linewidth=2, label="y_true", zorder=10)
    night_experts = [
        (f"{model}_Global",             "gray",    "-",  1.2, "Global (ref)"),
        (f"{model}_Night_Noise_Faible", "#FF00D9", "--", 1.2, "Night bruit faible (10%)"),
        (f"{model}_Night_Noise_Modere", "#88FF00", "--", 1.2, "Night bruit modéré (50%)"),
        (f"{model}_Night_Noise_Fort",   "#FF4500", "--", 1.2, "Night bruit fort (90%)"),
        (f"{model}_Night",              "#FFEE03", "-",  1.5, "Night masqué (actuel)"),
    ]
    for col, color, ls, lw, label in night_experts:
        if col in period_df.columns:
            ax_night.plot(hours, period_df[col].values, ls, color=color,
                          linewidth=lw, alpha=0.8, label=label)
    ax_night.set_title(f"Experts NUIT — Comparaison niveaux de bruit ({model})")
    ax_night.set_ylabel("Puissance (MW)")
    ax_night.grid(True, alpha=0.3)

    # Axe x : une date par jour
    tick_pos = [d * 24 for d in range(n_days)]
    tick_labels = [period_df["Date_Heure"].iloc[d * 24].strftime("%Y-%m-%d")
                   for d in range(n_days)]
    ax_night.set_xticks(tick_pos)
    ax_night.set_xticklabels(tick_labels, rotation=45)
    ax_night.set_xlabel("Date")

    # Légende enrichie avec les bandes jour/nuit
    from matplotlib.patches import Patch
    for ax in (ax_day, ax_night):
        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor='gold', alpha=0.2, label='Jour (6h–20h)'),
                    Patch(facecolor='midnightblue', alpha=0.2, label='Nuit (21h–5h)')]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

    date_start = period_df["Date_Heure"].iloc[0].strftime("%Y-%m-%d")
    date_end = period_df["Date_Heure"].iloc[-1].strftime("%Y-%m-%d")
    fig.suptitle(f"Comparaison experts jour/nuit bruités — {date_start} à {date_end}",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f"Data/plot_noise_day_night_{_ts}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot sauvegardé : Data/plot_noise_day_night_{_ts}.png")


# On récupère les noms des experts triés par performance
sorted_names = results.sort_values("nmae_%")["Exp_name"].tolist()
# On récupère les prédictions correspondantes
sorted_preds = [experts_preds_test[i] for i in results.sort_values("nmae_%").index]

# Tracer le jour 10 par exemple
#plot_day_ahead(28, y_test_24, sorted_preds, sorted_names, y_pred_agg_test, ref_capacity)
# On affiche sur 5 jours à partir du jour 10 les résultats

#plot_day_ahead_lasso(10, n_days=5)
# On affiche sur 5 jours à partir du jour 15 les résultats
plot_day_ahead_lasso(15, n_days=5, df_csv_path=f"Data/predictions_experts_{_ts}.csv")

# ---------------- Visualisation des experts saisons ----------------
def plot_season_experts(df_csv_path="Data/predictions_experts.csv", model="LGBM"):
    """Trace les 4 experts saisonniers vs y_true, 1 mois représentatif par saison."""
    df = pd.read_csv(df_csv_path, parse_dates=["Date_Heure"], sep=";")

    season_cols = {
        #"Spring (20 mar – 20 jun)": f"{model}_Spring",
        "Summer (21 jun – 22 sep)": f"{model}_Summer",
        #"Autumn (23 sep – 22 déc)": f"{model}_Autumn",
        "Winter (23 déc – 19 mar)": f"{model}_Winter",
    }
    season_cols = {k: v for k, v in season_cols.items() if v in df.columns}
    if not season_cols:
        print(f"Aucun expert saison trouvé pour le modèle {model}")
        return

    colors_season = {#"spring": "#77DD77",
                    "summer": "#FFD700",
                    #"autumn": "#CD853F",
                    "winter": "#87CEEB"}

    # Sélectionner 1 mois représentatif par saison
    md = df["Date_Heure"].dt.month * 100 + df["Date_Heure"].dt.day
    season_defs = [
        #("Printemps", "spring", (md >= 320) & (md <= 620)),
        ("Été",       "summer", (md >= 621) & (md <= 922)),
        #("Automne",   "autumn", (md >= 923) & (md <= 1222)),
        ("Hiver",     "winter", (md >= 1223) | (md <= 319)),
    ]

    panels = []
    month_size = 30 * 24
    for label, key, mask in season_defs:
        idx = df.index[mask].values
        if len(idx) == 0:
            continue
        # Découper en blocs contigus
        gaps = np.where(np.diff(idx) > 1)[0] + 1
        blocks = np.split(idx, gaps)
        for i, block in enumerate(blocks):
            if len(block) < 24:  # ignorer les blocs < 1 jour
                continue
            mid = len(block) // 2
            start = max(0, mid - month_size // 2)
            end = min(len(block), start + month_size)
            suffix = f" ({i+1})" if len(blocks) > 1 else ""
            panels.append((f"{label}{suffix}", key, df.iloc[block[start:end]]))
        
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(22, 5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (label, key, chunk_df) in zip(axes, panels):
        chunk_df = chunk_df.reset_index(drop=True)
        ax.axvspan(chunk_df["Date_Heure"].iloc[0], chunk_df["Date_Heure"].iloc[-1],
                   color=colors_season[key], alpha=0.12)

        ax.plot(chunk_df["Date_Heure"], chunk_df["y_true"],
                color="black", linewidth=1.5, label="y_true", zorder=10)
        for slabel, col in season_cols.items():
            ax.plot(chunk_df["Date_Heure"], chunk_df[col],
                    "--", linewidth=1, alpha=0.7, label=slabel)

        d0 = chunk_df["Date_Heure"].iloc[0].strftime("%Y-%m-%d")
        d1 = chunk_df["Date_Heure"].iloc[-1].strftime("%Y-%m-%d")
        ax.set_title(f"{label} — {d0} à {d1}")
        ax.set_ylabel("Puissance (MW)")
        ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Patch(facecolor=c, alpha=0.3, label=n.capitalize())
                for n, c in colors_season.items()]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)

    fig.suptitle(f"Experts Saisonniers ({model}) — 1 mois par saison", fontsize=14, y=1.01)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"Data/plot_experts_saisons_{_ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot sauvegardé : Data/plot_experts_saisons_{_ts}.png")

# ---------------- Visualisation des experts trimestres ----------------
def plot_trimester_experts(df_csv_path="Data/predictions_experts.csv", model="LGBM"):
    """Trace les 4 experts trimestriels vs y_true, 1 mois représentatif par trimestre."""
    df = pd.read_csv(df_csv_path, parse_dates=["Date_Heure"], sep=";")
    trim_info = {
        "Trim_Venteux_Fort (Nov,Déc,Jan)": (f"{model}_Trim_Venteux_Fort", [11, 12, 1], "#0095ff"),
        "Trim_Venteux (Fév,Mar,Oct)":      (f"{model}_Trim_Venteux",      [2, 3, 10],  "#2ca02c"),
        "Trim_Transition (Avr,Mai,Sep)":    (f"{model}_Trim_Transition",   [4, 5, 9],   "#ff7f0e"),
        "Trim_Calme (Jun,Jul,Aoû)":        (f"{model}_Trim_Calme",        [6, 7, 8],   "#d62728"),
    }
    trim_info = {k: v for k, v in trim_info.items() if v[0] in df.columns}
    if not trim_info:
        print(f"Aucun expert trimestre trouvé pour le modèle {model}")
        return

    all_months_colors = {}
    for _, (_, months, color) in trim_info.items():
        for m in months:
            all_months_colors[m] = color

    month_series = df["Date_Heure"].dt.month
    month_size = 30 * 24

    panels = []
    for label, (col, months, color) in trim_info.items():
        mask = month_series.isin(months)
        idx = df.index[mask].values
        if len(idx) == 0:
            continue
        gaps = np.where(np.diff(idx) > 1)[0] + 1
        blocks = np.split(idx, gaps)
        for i, block in enumerate(blocks):
            if len(block) < 24:
                continue
            mid = len(block) // 2
            start = max(0, mid - month_size // 2)
            end = min(len(block), start + month_size)
            suffix = f" ({i+1})" if len(blocks) > 1 else ""
            panels.append((f"{label}{suffix}", color, df.iloc[block[start:end]]))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(22, 5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (label, bg_color, chunk_df) in zip(axes, panels):
        chunk_df = chunk_df.reset_index(drop=True)
        ax.axvspan(chunk_df["Date_Heure"].iloc[0], chunk_df["Date_Heure"].iloc[-1],
                   color=bg_color, alpha=0.10)

        ax.plot(chunk_df["Date_Heure"], chunk_df["y_true"],
                color="black", linewidth=1.5, label="y_true", zorder=10)
        for tlabel, (col, _, color) in trim_info.items():
            ax.plot(chunk_df["Date_Heure"], chunk_df[col],
                    "--", color=color, linewidth=1, alpha=0.7, label=tlabel)

        d0 = chunk_df["Date_Heure"].iloc[0].strftime("%Y-%m-%d")
        d1 = chunk_df["Date_Heure"].iloc[-1].strftime("%Y-%m-%d")
        ax.set_title(f"{label} — {d0} à {d1}")
        ax.set_ylabel("Puissance (MW)")
        ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    handles, labels = axes[0].get_legend_handles_labels()
    for label, (_, _, color) in trim_info.items():
        handles.append(Patch(facecolor=color, alpha=0.25,
                             label=f"Zone {label.split('(')[0].strip()}"))
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)

    fig.suptitle(f"Experts Trimestriels ({model}) — 1 mois par trimestre", fontsize=14, y=1.01)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(f"Data/plot_experts_trimestres_{_ts}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot sauvegardé : Data/plot_experts_trimestres_{_ts}.png")


# Appels
csv_path = f"Data/predictions_experts_{_ts}.csv"
plot_season_experts(df_csv_path=csv_path, model="LGBM")
plot_trimester_experts(df_csv_path=csv_path, model="LGBM")
plot_day_night_noise_comparison(15, n_days=7, df_csv_path=csv_path, model="LGBM")

# ---------------- PARTIE QUI EVALUE LES DIFFERENTS EXPERTS ----------------

# ---------------- Diagnostic jour/nuit par expert ----------------
from sklearn.metrics import mean_absolute_error

df_diag = pd.read_csv(f"Data/predictions_experts_{_ts}.csv", parse_dates=["Date_Heure"], sep=";")
df_diag["hour"] = df_diag["Date_Heure"].dt.hour
df_diag["is_day"] = (df_diag["hour"] >= 6) & (df_diag["hour"] < 21)

# Visualisation uniquement des experts :
# LGBM_Day, LGBM_Night, Ridge_Day, Ridge_Night,
# RandomForest_Day, RandomForest_Night
expert_cols = [c for c in df_diag.columns if c not in ("Date_Heure", "y_true", "hour", "is_day")
               and ("Day" in c or "Night" in c)]

rows = []
for col in expert_cols:
    mae_day = mean_absolute_error(
        df_diag.loc[df_diag["is_day"], "y_true"],
        df_diag.loc[df_diag["is_day"], col]
    )
    mae_night = mean_absolute_error(
        df_diag.loc[~df_diag["is_day"], "y_true"],
        df_diag.loc[~df_diag["is_day"], col]
    )
    rows.append({"expert": col, "MAE_jour": round(mae_day, 2), "MAE_nuit": round(mae_night, 2)})

df_diag_result = pd.DataFrame(rows)
df_diag_result["meilleur_quand"] = np.where(
    df_diag_result["MAE_jour"] < df_diag_result["MAE_nuit"], "JOUR", "NUIT"
)
df_diag_result["model"] = df_diag_result["expert"].str.split("_", n=1).str[0]
df_diag_result.sort_values(by=["model", "MAE_jour"], ascending=True, inplace=True)
print("\n--- Diagnostic jour/nuit ---")
print(df_diag_result.drop(columns="model").to_string(index=False))

# ---------------- Diagnostic spécialisation par régime (vent, orientation, synoptique) ----------------
print("\n" + "="*80)
print("DIAGNOSTIC SPÉCIALISATION PAR RÉGIME")
print("="*80)

# Récupérer les prédictions de TOUS les experts sur le test set
df_all = pd.DataFrame({
    "Date_Heure": X_test["Date_Heure"].values,
    "y_true": y_test.values.flatten(),
})
for name, preds in zip(expert_names, experts_preds_test):
    df_all[name] = preds

# Reconstruire les masques sur X_test (même logique que data_engineering.py)
wind_norm_test = X_test["Wind_Norm"].values
q33 = X_test["Wind_Norm"].quantile(0.33)
q66 = X_test["Wind_Norm"].quantile(0.66)

mask_low  = wind_norm_test <= q33
mask_med  = (wind_norm_test > q33) & (wind_norm_test <= q66)
mask_high = wind_norm_test > q66

sin_dir = X_test["Wind_Dir_Meteo_sin"].values
cos_dir = X_test["Wind_Dir_Meteo_cos"].values
mask_NE = (sin_dir > 0) & (cos_dir > 0)
mask_NW = (sin_dir > 0) & (cos_dir <= 0)
mask_SE = (sin_dir <= 0) & (cos_dir > 0)
mask_SW = (sin_dir <= 0) & (cos_dir <= 0)

# --- 1. Diagnostic Wind Level (Low / Med / High) ---
print("\n--- Diagnostic Wind Level ---")
print(f"Seuils : q33={q33:.1f}, q66={q66:.1f}")
print(f"Heures Low={mask_low.sum()}, Med={mask_med.sum()}, High={mask_high.sum()}")

wind_level_experts = [c for c in df_all.columns if any(
    x in c for x in ["Wind_Low", "Wind_Med", "Wind_High", "Global_all_wind"]
) and c not in ("Date_Heure", "y_true")]

rows_wl = []
for col in wind_level_experts:
    mae_low = mean_absolute_error(df_all.loc[mask_low, "y_true"], df_all.loc[mask_low, col])
    mae_med = mean_absolute_error(df_all.loc[mask_med, "y_true"], df_all.loc[mask_med, col])
    mae_high = mean_absolute_error(df_all.loc[mask_high, "y_true"], df_all.loc[mask_high, col])
    best = ["Low", "Med", "High"][np.argmin([mae_low, mae_med, mae_high])]
    rows_wl.append({"expert": col, "MAE_low": round(mae_low, 2),
                     "MAE_med": round(mae_med, 2), "MAE_high": round(mae_high, 2),
                     "meilleur_quand": best})

# Ajouter Global comme référence
global_experts = [c for c in df_all.columns if "Global" in c and "all" not in c
                  and c not in ("Date_Heure", "y_true")]
for col in global_experts:
    mae_low = mean_absolute_error(df_all.loc[mask_low, "y_true"], df_all.loc[mask_low, col])
    mae_med = mean_absolute_error(df_all.loc[mask_med, "y_true"], df_all.loc[mask_med, col])
    mae_high = mean_absolute_error(df_all.loc[mask_high, "y_true"], df_all.loc[mask_high, col])
    best = ["Low", "Med", "High"][np.argmin([mae_low, mae_med, mae_high])]
    rows_wl.append({"expert": col + " (ref)", "MAE_low": round(mae_low, 2),
                     "MAE_med": round(mae_med, 2), "MAE_high": round(mae_high, 2),
                     "meilleur_quand": best})

df_wl = pd.DataFrame(rows_wl)
df_wl["model"] = df_wl["expert"].str.replace(r" \(ref\)$", "", regex=True).str.split("_", n=1).str[0]
df_wl.sort_values(by=["model", "MAE_low"], ascending=True, inplace=True)
print(df_wl.drop(columns="model").to_string(index=False))

# --- 2. Diagnostic Wind Orientation (NE / NW / SE / SW) ---
print("\n--- Diagnostic Wind Orientation ---")
print(f"Heures NE={mask_NE.sum()}, NW={mask_NW.sum()}, SE={mask_SE.sum()}, SW={mask_SW.sum()}")

orient_experts = [c for c in df_all.columns if any(
    x in c for x in ["Wind_orientation_NE", "Wind_orientation_NW",
                      "Wind_orientation_SE", "Wind_orientation_SW", "Global_all_orient"]
) and c not in ("Date_Heure", "y_true")]

rows_or = []
for col in orient_experts:
    mae_ne = mean_absolute_error(df_all.loc[mask_NE, "y_true"], df_all.loc[mask_NE, col])
    mae_nw = mean_absolute_error(df_all.loc[mask_NW, "y_true"], df_all.loc[mask_NW, col])
    mae_se = mean_absolute_error(df_all.loc[mask_SE, "y_true"], df_all.loc[mask_SE, col])
    mae_sw = mean_absolute_error(df_all.loc[mask_SW, "y_true"], df_all.loc[mask_SW, col])
    best = ["NE", "NW", "SE", "SW"][np.argmin([mae_ne, mae_nw, mae_se, mae_sw])]
    rows_or.append({"expert": col, "MAE_NE": round(mae_ne, 2), "MAE_NW": round(mae_nw, 2),
                     "MAE_SE": round(mae_se, 2), "MAE_SW": round(mae_sw, 2),
                     "meilleur_quand": best})

# Ajouter Global comme référence
for col in global_experts:
    mae_ne = mean_absolute_error(df_all.loc[mask_NE, "y_true"], df_all.loc[mask_NE, col])
    mae_nw = mean_absolute_error(df_all.loc[mask_NW, "y_true"], df_all.loc[mask_NW, col])
    mae_se = mean_absolute_error(df_all.loc[mask_SE, "y_true"], df_all.loc[mask_SE, col])
    mae_sw = mean_absolute_error(df_all.loc[mask_SW, "y_true"], df_all.loc[mask_SW, col])
    best = ["NE", "NW", "SE", "SW"][np.argmin([mae_ne, mae_nw, mae_se, mae_sw])]
    rows_or.append({"expert": col + " (ref)", "MAE_NE": round(mae_ne, 2), "MAE_NW": round(mae_nw, 2),
                     "MAE_SE": round(mae_se, 2), "MAE_SW": round(mae_sw, 2),
                     "meilleur_quand": best})

df_or = pd.DataFrame(rows_or)
df_or["model"] = df_or["expert"].str.replace(r" \(ref\)$", "", regex=True).str.split("_", n=1).str[0]
df_or.sort_values(by=["model", "MAE_NE"], ascending=True, inplace=True)
print(df_or.drop(columns="model").to_string(index=False))

# --- 3. Diagnostic Synoptique vs Global ---
print("\n--- Diagnostic Synoptique vs Global ---")
syno_experts = [c for c in df_all.columns if "Synoptique" in c and c not in ("Date_Heure", "y_true")]

rows_syn = []
for col in syno_experts + global_experts:
    mae_total = mean_absolute_error(df_all["y_true"], df_all[col])
    label = col if col in syno_experts else col + " (ref)"
    rows_syn.append({"expert": label, "MAE_total": round(mae_total, 2)})

df_syn = pd.DataFrame(rows_syn)
df_syn["model"] = df_syn["expert"].str.replace(r" \(ref\)$", "", regex=True).str.split("_", n=1).str[0]
df_syn.sort_values(by=["model", "MAE_total"], ascending=True, inplace=True)
print(df_syn.drop(columns="model").to_string(index=False))

# --- 4. Diagnostic experts bruités : erreur sur période "off"
print("\n--- Diagnostic experts bruités : erreur sur période OFF ---")

noise_experts_day = [
    (f"{m}_Global",           "Global (ref)",       "nuit")
    for m in ["LGBM", "RandomForest", "Ridge"]
] + [
    (f"{m}_Day_Noise_Faible", "Day bruit 10%",      "nuit")
    for m in ["LGBM", "RandomForest", "Ridge"]
] + [
    (f"{m}_Day_Noise_Modere", "Day bruit 50%",      "nuit")
    for m in ["LGBM", "RandomForest", "Ridge"]
] + [
    (f"{m}_Day_Noise_Fort",   "Day bruit 90%",      "nuit")
    for m in ["LGBM", "RandomForest", "Ridge"]
] + [
    (f"{m}_Day",              "Day masqué (actuel)", "nuit")
    for m in ["LGBM", "RandomForest", "Ridge"]
]

noise_experts_night = [
    (f"{m}_Global",             "Global (ref)",         "jour")
    for m in ["LGBM", "RandomForest", "Ridge"]
] + [
    (f"{m}_Night_Noise_Faible", "Night bruit 10%",      "jour")
    for m in ["LGBM", "RandomForest", "Ridge"]
] + [
    (f"{m}_Night_Noise_Modere", "Night bruit 50%",      "jour")
    for m in ["LGBM", "RandomForest", "Ridge"]
] + [
    (f"{m}_Night_Noise_Fort",   "Night bruit 90%",      "jour")
    for m in ["LGBM", "RandomForest", "Ridge"]
] + [
    (f"{m}_Night",              "Night masqué (actuel)", "jour")
    for m in ["LGBM", "RandomForest", "Ridge"]
]

rows_noise = []
for col, label, off_period in noise_experts_day + noise_experts_night:
    if col not in df_diag.columns:
        continue
    if off_period == "nuit":
        mask_off = ~df_diag["is_day"]
    else:
        mask_off = df_diag["is_day"]

    mae_off = mean_absolute_error(
        df_diag.loc[mask_off, "y_true"],
        df_diag.loc[mask_off, col]
    )
    nmae_off = (mae_off / ref_capacity) * 100

    rows_noise.append({
        "expert": col,
        "type": label,
        "période_off": off_period,
        "MAE_off": round(mae_off, 2),
        "nMAE_off_%": round(nmae_off, 2),
    })

df_noise = pd.DataFrame(rows_noise)

# Calculer le delta % par rapport à l'expert Global pour chaque modèle x période_off
# = là ou il n'est pas spécialisé

for idx, row in df_noise.iterrows():
    model = row["expert"].split("_")[0]
    off = row["période_off"]
    ref_row = df_noise[
        (df_noise["expert"] == f"{model}_Global") &
        (df_noise["période_off"] == off)
    ]
    if not ref_row.empty:
        ref_nmae = ref_row["nMAE_off_%"].values[0]
        delta = ((row["nMAE_off_%"] - ref_nmae) / ref_nmae) * 100 if ref_nmae > 0 else 0
        df_noise.loc[idx, "Δ%_vs_Global"] = round(delta, 1)

df_noise.sort_values(by=["période_off", "expert"], inplace=True)
print(df_noise.to_string(index=False))

# Fermeture du fichier log
_log_file.close()
sys.stdout = sys.__stdout__
print(f"Log complet sauvegardé : {_log_path}")