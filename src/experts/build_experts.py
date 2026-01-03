import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.experts.expert_ElasticNet import ElasticNetExpert
from src.experts.expert_LGBM import LGBMExpert
from src.experts.expert_RandomForest import RandomForestExpert
from src.experts.expert_Ridge import RidgeExpert


DATA_PATH = "data/processed_data/data_engineering_belgique.csv"
OUTPUT_DIR = "data/experts"

N_BACKTEST = 5000
SAVE_EVERY = 100     

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load data
Data_clean_Belgique = pd.read_csv(DATA_PATH)

split_start = len(Data_clean_Belgique) - 24 - N_BACKTEST


# init results
results = pd.DataFrame({
    "Date_Heure": Data_clean_Belgique["Date_Heure"]
        .iloc[split_start:split_start + N_BACKTEST]
        .reset_index(drop=True),
    "y_true": Data_clean_Belgique["Eolien_MW"]
        .iloc[split_start:split_start + N_BACKTEST]
        .reset_index(drop=True)
})


def rolling_forecast(model, model_name, use_scaler=False):
    """
    Backtesting rolling window 1-step ahead
    Sauvegarde incrémentale automatique
    """

    predictions = []

    for i in range(N_BACKTEST):
        print(f"[{model_name}] step {i + 1}/{N_BACKTEST}")

        split_point = split_start + i
        train_df = Data_clean_Belgique.iloc[:split_point]

        X_train = train_df[model.features]
        y_train = train_df["Eolien_MW"]

        # ===== scaling (re fit at each iter) =====
        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train, columns=model.features)

        # ===== train =====
        model.fit(X_train, y_train)

        # ===== prediction =====
        X_next = Data_clean_Belgique[model.features].iloc[[split_point]]

        if use_scaler:
            X_next = scaler.transform(X_next)
            X_next = pd.DataFrame(X_next, columns=model.features)

        y_pred = model.predict(X_next)[0]
        predictions.append(y_pred)

        # ===== save =====
        results.loc[i, model_name] = y_pred

        if (i + 1) % SAVE_EVERY == 0:
            results.to_pickle(f"{OUTPUT_DIR}/experts_partial.pkl")

    return predictions


# experts
print("\n=== RandomForest ===")
results["randomforest"] = rolling_forecast(
    RandomForestExpert(),
    model_name="randomforest",
    use_scaler=False
)

print("\n=== LightGBM ===")
results["lgbm"] = rolling_forecast(
    LGBMExpert(),
    model_name="lgbm",
    use_scaler=False
)

print("\n=== ElasticNet ===")
results["elasticnet"] = rolling_forecast(
    ElasticNetExpert(),
    model_name="elasticnet",
    use_scaler=True
)


# save
results.to_csv(f"{OUTPUT_DIR}/experts.csv", index=False)
results.to_pickle(f"{OUTPUT_DIR}/experts.pkl")

print(f"\n Results saved in {OUTPUT_DIR}/")


# metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2


metrics_table = pd.DataFrame(
    columns=["MAE", "RMSE", "MAPE (%)", "R2"],
    index=["randomforest", "lgbm", "elasticnet"]
)

for model_name in metrics_table.index:
    metrics_table.loc[model_name] = compute_metrics(
        results["y_true"],
        results[model_name]
    )

metrics_table.to_csv(f"{OUTPUT_DIR}/experts_metrics.csv")
print("\n=== MÉTRIQUES ===")
print(metrics_table)


# plot
results["Date_Heure"] = pd.to_datetime(results["Date_Heure"])
results = results.set_index("Date_Heure")

plt.figure(figsize=(14, 6))
plt.plot(results.index, results["y_true"], label="True", linewidth=2)
plt.plot(results.index, results["randomforest"], label="RandomForest", alpha=0.8)
plt.plot(results.index, results["lgbm"], label="LGBM", alpha=0.8)
plt.plot(results.index, results["elasticnet"], label="ElasticNet", alpha=0.8)

plt.xlabel("Date / Heure")
plt.ylabel("Production éolienne (MW)")
plt.title("Prévision 1h ahead – Backtesting rolling")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
