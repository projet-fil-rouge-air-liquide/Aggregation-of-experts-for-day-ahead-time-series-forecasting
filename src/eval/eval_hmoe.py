import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.opera.hmoe import prepare_features, train_hmoe, predict_hmoe

def main():
    df = pd.read_csv("data/experts/experts.csv")

    targets, experts, regime_features, valid_idx = prepare_features(df)

    history = 4500
    test_step = 10
    model = "FTRL"

    preds_hmoe, preds_rf, preds_lgbm, preds_en = [], [], [], []
    y_true_list = []

    for t in tqdm(range(history, len(valid_idx) - 1, test_step), desc="Evaluation"):
        idx_train = valid_idx[t - history : t]
        idx_test = valid_idx[t + 1]

        # train
        hmoe = train_hmoe(df, idx_train, model)

        # predict
        y_pred = predict_hmoe(hmoe, df, idx_test)
        y_true = targets.loc[idx_test]

        preds_hmoe.append(y_pred)
        preds_rf.append(experts.loc[idx_test, "randomforest"])
        preds_lgbm.append(experts.loc[idx_test, "lgbm"])
        preds_en.append(experts.loc[idx_test, "elasticnet"])
        y_true_list.append(y_true)

    # METRICS
    y_true_arr = np.array(y_true_list)

    results = []

    for name, y_pred in {
        "HMoE": preds_hmoe,
        "RF": preds_rf,
        "LGBM": preds_lgbm,
        "EN": preds_en,
    }.items():

        y_pred_arr = np.array(y_pred)
        err = y_true_arr - y_pred_arr
        abs_err = np.abs(err)

        results.append({
            "model": name,

            # Central tendency
            "MAE_mean": abs_err.mean(),
            "MAE_std": abs_err.std(),

            # Tail risk
            "MAE_p90": np.quantile(abs_err, 0.90),
            "MAE_p95": np.quantile(abs_err, 0.95),
            "MAE_max": abs_err.max(),

            # interquartile range
            "MAE_iqr": np.quantile(abs_err, 0.75) - np.quantile(abs_err, 0.25),
        })
        
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/eval/eval_hmoe_vs_experts.csv", index=False)

    print(results_df)


if __name__ == "__main__":
    main()