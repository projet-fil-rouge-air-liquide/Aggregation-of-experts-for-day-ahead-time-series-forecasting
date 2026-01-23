import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.opera.mixture import Mixture

from utils.eval_utils import compute_metrics

df = pd.read_csv("data/experts/experts.csv")

targets = df["y_true"]
experts = df[["randomforest", "lgbm", "elasticnet"]]

train_size = int(0.95 * len(df))
prediction_window = 24

train_targets = targets.iloc[:train_size]
train_experts = experts.iloc[:train_size]

test_targets = targets.iloc[train_size:]
test_experts = experts.iloc[train_size:]

expert_names = experts.columns.tolist()

prediction_window = 24

parameters = {
    "fun_reg": lambda w: 0.1 * np.sum(w**2),
    "fun_reg_grad": lambda w: 0.2 * w,
    "constraints": [{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
    "options": {"maxiter": 50},
}


awake_train = np.ones((len(train_experts), train_experts.shape[1]))

moe = Mixture(
    y=train_targets,
    experts=train_experts,
    awake=awake_train,
    model="FTRL",
    loss_type="mse",
    parameters=parameters
)

metrics = []

for i in range(0, len(test_targets) - prediction_window + 1):
    y_true_window = test_targets.iloc[i:i + prediction_window].values
    experts_window = test_experts.iloc[i:i + prediction_window]

    # ----- MoE prediction -----
    y_pred_moe = moe.predict(experts_window)
    mae, rmse, r2 = compute_metrics(y_true_window, y_pred_moe)

    metrics.append({
        "model": "MoE",
        "window_start": train_size + i,
        "window_end": train_size + i + prediction_window,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # expert
    for expert in expert_names:
        y_pred_exp = experts_window[expert].values
        mae, rmse, r2 = compute_metrics(y_true_window, y_pred_exp)

        metrics.append({
            "model": expert,
            "window_start": train_size + i,
            "window_end": train_size + i + prediction_window,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("data/eval/moe_vs_experts.csv", index=False)

summary = (
    metrics_df
    .groupby("model")[["MAE", "RMSE", "R2"]]
    .mean()
    .reset_index()
)

print(summary)

summary.to_csv("data/eval/eval_moe_vs_experts_summary.csv", index=False)