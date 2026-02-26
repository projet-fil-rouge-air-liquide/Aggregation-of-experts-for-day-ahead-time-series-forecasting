import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.opera.hmoe import prepare_features, train_hmoe, predict_hmoe

def compute_metrics(y_true, y_pred):
    """Calcule les métriques MAE, RMSE et MAPE avec moyenne, variance et p95."""
    err = y_true - y_pred
    abs_err = np.abs(err)
    sq_err = err ** 2
    mape = np.abs(err / np.clip(y_true, 1e-8, None))

    # MAE
    mae_mean = abs_err.mean()
    mae_var = abs_err.var()
    mae_p95 = np.quantile(abs_err, 0.95)

    # RMSE
    rmse_mean = np.sqrt(sq_err.mean())
    rmse_var = sq_err.var()
    rmse_p95 = np.sqrt(np.quantile(sq_err, 0.95))

    # MAPE (%)
    mape_mean = mape.mean() * 100
    mape_var = mape.var() * 100
    mape_p95 = np.quantile(mape, 0.95) * 100

    return {
        "MAE_mean": mae_mean, "MAE_var": mae_var, "MAE_p95": mae_p95,
        "RMSE_mean": rmse_mean, "RMSE_var": rmse_var, "RMSE_p95": rmse_p95,
        "MAPE_mean": mape_mean, "MAPE_var": mape_var, "MAPE_p95": mape_p95,
    }

def main():
    df = pd.read_csv("data/experts/experts_feat.csv")

    targets, experts, regime_features, valid_idx = prepare_features(df)

    history = 6500
    test_step = 3 # 5
    model = "FTRL"

    # stockage des erreurs
    errors = {
        "HMoE": [],
        "RF": [],
        "LGBM": [],
        "EN": [],
    }

    mape_errors = {
        "HMoE": [],
        "RF": [],
        "LGBM": [],
        "EN": [],
    }

    for t in tqdm(range(history, len(valid_idx) - 1, test_step), desc="Evaluation"):
        idx_train = valid_idx[t - history : t]
        idx_test = valid_idx[t + 1]

        hmoe = train_hmoe(df, idx_train, model)

        y_true = targets.loc[idx_test]

        preds = {
            "HMoE": predict_hmoe(hmoe, df, idx_test),
            "RF": experts.loc[idx_test, "randomforest"],
            "LGBM": experts.loc[idx_test, "lgbm"],
            "EN": experts.loc[idx_test, "elasticnet"],
        }

        for name, y_pred in preds.items():
            err = y_true - y_pred
            errors[name].append(err)
            mape_errors[name].append(
                np.abs(err / np.clip(y_true, 1e-8, None))
            )

    results = []

    for name in errors.keys():
        err = np.array(errors[name])
        abs_err = np.abs(err)
        sq_err = err ** 2
        mape = np.array(mape_errors[name])

        results.append({
            "model": name,

            # MAE
            "MAE_mean": abs_err.mean(),
            "MAE_var": abs_err.var(),
            "MAE_p95": np.quantile(abs_err, 0.95),

            # RMSE
            "RMSE_mean": np.sqrt(sq_err.mean()),
            "RMSE_var": sq_err.var(),
            "RMSE_p95": np.sqrt(np.quantile(sq_err, 0.95)),

            # MAPE
            "MAPE_mean": mape.mean() * 100,
            "MAPE_var": mape.var() * 100,
            "MAPE_p95": np.quantile(mape, 0.95) * 100,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/eval/eval_hmoe_vs_experts.csv", index=False)

    print(results_df)


if __name__ == "__main__":
    main()