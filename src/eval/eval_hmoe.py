import pandas as pd
import numpy as np
from tqdm import tqdm
from src.opera.mixture import HierarchicalHorizonOPERA, RegimeGate
from utils.eval_utils import compute_metrics


df = pd.read_csv("data/experts/experts.csv")

targets = df["y_true"]
experts = df[["randomforest", "lgbm", "elasticnet"]]
regime_features = df[["ret_1", "ret_24", "vol_24", "mom_24", "hour_sin", "hour_cos"]]

valid_idx = regime_features.dropna().index
df = df.loc[valid_idx]
targets = targets.loc[valid_idx]
experts = experts.loc[valid_idx]
regime_features = regime_features.loc[valid_idx]


parameters = {"eta": 0.05, "l1": 0.0, "l2": 0.01}
horizons = [1]
regimes = ["bull", "bear"]
prediction_window = 24

train_size = int(0.8 * len(df))

train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]


hmod = HierarchicalHorizonOPERA(
    y=train_df["y_true"].values,
    experts=train_df[["randomforest", "lgbm", "elasticnet"]],
    regimes=regimes,
    horizons=horizons,
    regime_gate=RegimeGate(
        n_regimes=len(regimes),
        lr=0.1
    ),
    model="FTRL",
    loss_type="mse",
    parameters=parameters,
)


N_train = len(train_df) - max(horizons)

for t in range(N_train):
    expert_preds_t = {
        h: train_df[["randomforest", "lgbm", "elasticnet"]].iloc[[t]]
        for h in horizons
    }
    y_true_t = {
        h: train_df["y_true"].iloc[t + h]
        for h in horizons
    }

    hmod.update(
        expert_preds=expert_preds_t,
        y_true=y_true_t,
        regime_features=regime_features.iloc[t].values,
        regime_label=None
    )


metrics_df = []

for i in tqdm(range(len(test_df) - prediction_window), desc="Evaluation"):
    y_true_window = []
    y_pred_window = []

    for t in range(prediction_window):
        idx = i + t

        expert_preds_t = {
            h: test_df[["randomforest", "lgbm", "elasticnet"]].iloc[[idx]]
            for h in horizons
        }

        y_pred_t = hmod.predict(
            expert_preds=expert_preds_t,
            regime_features=regime_features.iloc[train_size + idx].values
        )[1]

        y_true_t = test_df["y_true"].iloc[idx]

        y_pred_window.append(float(np.squeeze(y_pred_t)))
        y_true_window.append(float(y_true_t))

        # Update online
        hmod.update(
            expert_preds=expert_preds_t,
            y_true={1: y_true_t},
            regime_features=regime_features.iloc[train_size + idx].values,
            regime_label=None
        )

    mae, rmse, r2 = compute_metrics(y_true_window, y_pred_window)

    metrics_df.append({
        "window_start": train_size + i,
        "window_end": train_size + i + prediction_window,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

metrics_df = pd.DataFrame(metrics_df)

metrics_df.to_csv(
    "data/eval/eval_hmoe.csv",
    index=False
)

mean_mae = metrics_df["MAE"].mean()
mean_rmse = metrics_df["RMSE"].mean()
mean_r2 = metrics_df["R2"].mean()

print(f"MAE : {mean_mae:.4f}")
print(f"RMSE : {mean_rmse:.4f}")
print(f"R² : {mean_r2:.4f}")