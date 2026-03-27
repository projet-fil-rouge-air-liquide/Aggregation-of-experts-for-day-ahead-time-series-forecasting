import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.opera.hmoe import prepare_features, train_hmoe, predict_hmoe

def main():
    df = pd.read_csv("data/experts/predictions_experts_spe_feat_10000.csv")

    targets, experts, regime_features, valid_idx = prepare_features(df)

    history = 5000 # 5000
    test_step = 20
    model = "BOA" # MLpol, MLprod, BOA, FTRL
    context = ("daynight") # {} ; ("trend", "updown", "wind", "volatility", "daynight")

    if context=={}:
        method="MoE"
    else:
        method=f"HMoE_{context}"

    errors = {name: [] for name in ["HMoE", "RF", "LGBM", "Ridge"]}
    mape_errors = {name: [] for name in ["HMoE", "RF", "LGBM", "Ridge"]}

    all_y_true = []
    all_preds = {name: [] for name in ["HMoE", "RF", "LGBM", "Ridge"]}

    for t in tqdm(range(history, len(valid_idx) - 1, test_step), desc="Evaluation"):
        idx_train = valid_idx[t - history : t]
        idx_test = valid_idx[t + 1]

        hmoe = train_hmoe(df, idx_train, model, context)

        y_true = targets.loc[idx_test]

        preds = {
            "HMoE": predict_hmoe(hmoe, df, idx_test),
            "RF": experts.loc[idx_test, "RandomForest_Global"],
            "LGBM": experts.loc[idx_test, "LGBM_Global"],
            "Ridge": experts.loc[idx_test, "Ridge_Global"],
        }

        all_y_true.append(y_true)

        for name, y_pred in preds.items():
            all_preds[name].append(y_pred)

            err = y_true - y_pred
            errors[name].append(err)

            mape_errors[name].append(
                np.abs(err / np.clip(y_true, 1e-8, None))
            )

    results = []
    y_true_all = np.array(all_y_true)

    for name in errors.keys():
        err = np.array(errors[name])
        abs_err = np.abs(err)
        sq_err = err ** 2

        mape = np.abs(err) / np.maximum(np.abs(y_true), 1e-8)

        y_pred_all = np.array(all_preds[name])

        ss_res = np.sum((y_true_all - y_pred_all) ** 2)
        ss_tot = np.sum((y_true_all - y_true_all.mean()) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        results.append({
            "model": name,

            # MAE
            "MAE_mean": abs_err.mean(),
            "MAE_var": abs_err.var(),
            "MAE_p95": np.quantile(abs_err, 0.95),

            # RMSE
            "RMSE_mean": np.sqrt(sq_err.mean()),
            "RMSE_var": np.sqrt(sq_err.var()),
            "RMSE_p95": np.sqrt(np.quantile(sq_err, 0.95)),

            # MAPE
            "MAPE_mean": mape.mean() * 100,
            "MAPE_var": mape.var() * 100,
            "MAPE_p95": np.quantile(mape, 0.95) * 100,

            # R²
            "R2": r2,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"data/eval/{method}_hist-{history}_{model}.csv", index=False)

    print(results_df)


if __name__ == "__main__":
    main()