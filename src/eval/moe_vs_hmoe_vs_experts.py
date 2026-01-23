import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.eval_utils import compute_metrics
from src.experts.expert_ElasticNet import ElasticNetExpert
from src.experts.expert_LGBM import LGBMExpert
from src.experts.expert_RandomForest import RandomForestExpert
from src.opera.mixture import Mixture
from src.opera.mixture import HierarchicalHorizonOPERA, RegimeGate


df = pd.read_csv("data/processed_data/data_engineering_belgique_regime.csv")
experts_df = pd.read_csv("data/experts/experts.csv")

target_col = "Eolien_MW"
date_col = "Date_Heure"

prediction_window = 24

features = [
    c for c in df.columns
    if c not in [target_col, date_col]
       and pd.api.types.is_numeric_dtype(df[c])
]

'''
Recursive online forecast for ONE window - 
similar with recursive_forecast in prediction_for_24h.py
'''
def recursive_forecast_window(
    df,
    start_idx,
    expert,
    features,
    prediction_window=24,
    use_scaler=False
):
    """
    Recursive 24h forecast starting at start_idx
    """
    df_work = df.copy()
    preds = []

    if use_scaler:
        scaler = StandardScaler()
        scaler.fit(df_work[features].iloc[:start_idx])

    for step in range(prediction_window):

        split_point = start_idx + step
        train_df = df_work.iloc[:split_point]

        X_train = train_df[features]
        y_train = train_df[target_col]

        if use_scaler:
            X_train = pd.DataFrame(
                scaler.transform(X_train),
                columns=features
            )

        # Train expert
        expert.fit(X_train, y_train)

        # Predict next step
        X_next = df_work[features].iloc[[split_point]]

        if use_scaler:
            X_next = pd.DataFrame(
                scaler.transform(X_next),
                columns=features
            )

        y_pred = expert.predict(X_next)[0]
        preds.append(y_pred)

        # Inject prediction
        df_work.loc[split_point, target_col] = y_pred

    return np.array(preds)

# Experts configuration
experts = {
    "randomforest": (RandomForestExpert, False),
    "lgbm": (LGBMExpert, False),
    "elasticnet": (ElasticNetExpert, True),
}

# --- MoE ---
parameters_moe = {
    "fun_reg": lambda w: 0.1 * np.sum(w**2),
    "fun_reg_grad": lambda w: 0.2 * w,
    "constraints": [{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
    "options": {"maxiter": 50},
}

expert_cols = ["randomforest", "lgbm", "elasticnet"]

y_moe = experts_df["y_true"]
X_moe = experts_df[expert_cols]

train_size_expert = len(experts_df)

y_train_moe = y_moe.iloc[:train_size_expert]
X_train_moe = X_moe.iloc[:train_size_expert]
awake_train = np.ones((len(X_train_moe), X_train_moe.shape[1]))

moe = Mixture(
    y=y_train_moe,
    experts=X_train_moe,
    awake=awake_train,
    model="FTRL",
    loss_type="mse",
    parameters=parameters_moe
)

# --- HMoE ---
parameters_hmoe = {"eta": 0.05, "l1": 0.0, "l2": 0.01}
horizons = [1]
regimes = ["bull", "bear"]

hmod = HierarchicalHorizonOPERA(
    y=y_train_moe.values,
    experts=X_train_moe,
    regimes=regimes,
    horizons=horizons,
    regime_gate=RegimeGate(
        n_regimes=len(regimes),
        lr=0.1
    ),
    model="FTRL",
    loss_type="mse",
    parameters=parameters_hmoe,
)


# regime
regime_features_cols = ["ret_1", "ret_24", "vol_24", "mom_24", "hour_sin", "hour_cos"]

regime_features = df[regime_features_cols].copy()


# Sliding window evaluation
metrics = []
step = prediction_window
train_size = int(0.9 * len(df)) 
n_test_windows = len(df) - train_size - prediction_window + 1

for i in tqdm(range(0, n_test_windows, step), desc="Eval"):

    window_start = train_size + i
    window_end = window_start + prediction_window

    if window_end > len(df):
        break

    y_true = df[target_col].iloc[window_start:window_end].values

    # Experts recursiv
    expert_preds_24h = {}

    for name, (ExpertClass, use_scaler) in experts.items():

        expert = ExpertClass()

        y_pred = recursive_forecast_window(
            df=df,
            start_idx=window_start,
            expert=expert,
            features=features,
            prediction_window=prediction_window,
            use_scaler=use_scaler
        )

        expert_preds_24h[name] = y_pred

        mae, rmse, r2 = compute_metrics(y_true, y_pred)

        metrics.append({
            "model": name,
            "window_start": window_start,
            "window_end": window_end,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    # MoE
    experts_matrix = pd.DataFrame(expert_preds_24h)

    y_pred_moe = moe.predict(experts_matrix)

    mae, rmse, r2 = compute_metrics(y_true, y_pred_moe)

    metrics.append({
        "model": "MoE",
        "window_start": window_start,
        "window_end": window_end,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # HMoE
    y_true_hmoe = []
    y_pred_hmoe = []
    for t in range(prediction_window):
        expert_preds_t = {
            1: experts_matrix.iloc[[t]]
        }

        regime_t = regime_features.iloc[window_start + t].values

        y_pred_t = hmod.predict(
            expert_preds=expert_preds_t,
            regime_features=regime_t
        )[1]

        y_true_t = y_true[t]

        y_pred_hmoe.append(float(np.squeeze(y_pred_t)))
        y_true_hmoe.append(float(y_true_t))

        hmod.update(
            expert_preds=expert_preds_t,
            y_true={1: y_true_t},
            regime_features=regime_t,
            regime_label=None
        )

    mae, rmse, r2 = compute_metrics(y_true_hmoe, y_pred_hmoe)

    metrics.append({
        "model": "HMoE",
        "window_start": window_start,
        "window_end": window_end,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# Save results
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("data/eval/eval_experts_recursive_24h.csv", index=False)

summary = (
    metrics_df
    .groupby("model")[["MAE", "RMSE", "R2"]]
    .mean()
    .reset_index()
)

print(summary)

summary.to_csv(
    "data/eval/eval_experts_recursive_24h_summary.csv",
    index=False
)
