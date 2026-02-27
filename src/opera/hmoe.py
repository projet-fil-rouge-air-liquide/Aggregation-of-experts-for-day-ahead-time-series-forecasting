import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.opera.mixture import HMoE
from src.opera.regime import Regime, SoftmaxGate, WindRegime, TrendRegime


# DEBUG / ANALYSIS
def print_expert_weights(hmoe, expert_names):
    for regime_tuple, mixture in hmoe.experts_by_regime.items():
        print(f"\nRegime {regime_tuple}")
        w = mixture.w
        for name, wi in zip(expert_names, w):
            print(f"{name:15s} | w = {wi:.4f}")

def print_regime_probs(hmoe, regime_features_t):
    print("\n=== REGIME PROBABILITIES ===")
    for name, context in hmoe.regime_context.items():
        p = context.predict(regime_features_t[name])
        for r, pr in zip(context.regimes, p):
            print(f"{name:6s} | {r:5s} = {pr:.3f}")

def prepare_features(df):
    targets = df["y_true"]

    experts = df[["randomforest", "lgbm", "elasticnet"]]

    regime_features = {
    "trend": df[[
        "trend_strength",
        "mom_24",
        "mom_48",
        "vol_24",
    ]],
    "wind": df[[
        "Wind_Norm",
        "Wind_mean_3h",
        "Wind_Norm_lag_1h", 
        "Wind_Norm_lag_24h"
    ]]
    }

    valid_idx = (
        targets.dropna().index
        .intersection(experts.dropna().index)
        .intersection(regime_features["trend"].dropna().index)
        .intersection(regime_features["wind"].dropna().index)
    )

    return targets, experts, regime_features, valid_idx

def train_hmoe(df, idx_train, model):
    targets, experts, regime_features, _ = prepare_features(df)

    y_train = targets.loc[idx_train]
    X_train = experts.loc[idx_train]
    regime_train = {k: v.loc[idx_train] for k, v in regime_features.items()}

    trend_regime = Regime(
        name="trend",
        regimes=["bull", "bear"],
        gate=SoftmaxGate(2),
        prior=TrendRegime(trend_idx=0),
    )

    wind_std = np.std(regime_features["wind"].iloc[idx_train, 0].values) # std for wind_norm
    wind_mean = np.mean(regime_features["wind"].iloc[idx_train, 0].values)
    low_th = wind_mean - wind_std/2
    high_th = wind_mean + wind_std/2

    wind_regime = Regime(
        name="wind",
        regimes=["low", "high"],
        gate=SoftmaxGate(2),
        prior=WindRegime(wind_feature_idx=0, wind_mean=wind_mean, wind_std=wind_std),
    )
    if model=="FTRL":
        hmoe = HMoE(
            y=y_train,
            experts=X_train,
            regime_context={
                "trend": trend_regime,
                "wind": wind_regime,
            },
            model="FTRL",
            loss_type="mse",
            parameters={"eta": 0.05, "l1": 0.0, "l2": 0.01},
        )
    elif model in ["BOA", "MLprod", "MLpol"]:
        hmoe = HMoE(
            y=y_train,
            experts=X_train,
            regime_context={
                "trend": trend_regime,
                "wind": wind_regime,
            },
            model=model,
            loss_type="mse",
        )
    else:
        raise ValueError(
            "model must be one of ['FTRL', 'MLpol', 'MLprod', 'BOA']"
        )

    for i in tqdm(range(len(idx_train)), desc="Train"):
        regime_t = {
            name: regime_train[name].iloc[i].values
            for name in regime_train
        }

        hmoe.update(
            expert_preds=X_train.iloc[[i]],
            y_true=y_train.iloc[i],
            regime_features=regime_t,
        )

    return hmoe

def predict_hmoe(hmoe, df, idx_test):
    _, experts, regime_features, _ = prepare_features(df)

    expert_t = experts.loc[[idx_test]]
    regime_t = {
        name: feats.loc[idx_test].values
        for name, feats in regime_features.items()
    }

    return hmoe.predict(
        expert_preds=expert_t,
        regime_features=regime_t,
    ).item()

def rolling_forecast_24h(hmoe, df_last24, experts, regime_features):
    preds = []

    for idx in df_last24.index:
        expert_t = experts.loc[[idx]]
        regime_t = {
            name: feats.loc[idx].values
            for name, feats in regime_features.items()
        }

        y_pred = hmoe.predict(
            expert_preds=expert_t,
            regime_features=regime_t,
        ).item()

        preds.append(y_pred)

    return preds

def plot_24h_forecast(df_last24, experts_last24, y_pred, forecast):
    plt.figure(figsize=(15, 7))

    plt.plot(
        df_last24["Date_Heure"],
        df_last24["y_true"],
        label="y_true",
        lw=2.5,
    )

    for col in experts_last24.columns:
        plt.plot(
            df_last24["Date_Heure"],
            experts_last24[col],
            "--",
            alpha=0.6,
            label=col,
        )

    plt.plot(
        df_last24["Date_Heure"],
        y_pred,
        lw=2.5,
        label="HMoE prediction",
    )

    plt.title(f"{forecast}h forecast – HMoE (trend + wind)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def extract_regime_probs(hmoe, df, regime_features):
    probs_regimes = []
    for idx in df.index:
        row = {"index": idx}
        for name, context in hmoe.regime_context.items():
            probs = context.predict(
                regime_features[name].loc[idx].values
            )
            for regime, p in zip(context.regimes, probs):
                row[f"{name}_{regime}"] = p

        probs_regimes.append(row)

    probs_df = pd.DataFrame(probs_regimes).set_index("index")
    return probs_df


def plot_regime_probs(probs_df, df_slice):
    def minmax_norm(s):
        return (s - s.min()) / (s.max() - s.min())

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))

    axes[0].plot(df_slice["Date_Heure"], minmax_norm(df_slice["trend_strength"]), label="Trend Strength")
    axes[0].plot(df_slice["Date_Heure"], minmax_norm(df_slice["mom_48"]), label="Momentum 48h")
    axes[0].plot(df_slice["Date_Heure"], minmax_norm(df_slice["mom_24"]), label="Momentum 24h")
    axes[0].plot(df_slice["Date_Heure"], minmax_norm(df_slice["vol_24"]), label="Volatility 24h")
    axes[0].set_ylim(0, 1)

    axes[1].plot(df_slice["Date_Heure"], probs_df["trend_bull"], label="Bull", color="g")
    axes[1].plot(df_slice["Date_Heure"], probs_df["trend_bear"], label="Bear", color="b")

    axes[2].plot(df_slice["Date_Heure"], probs_df["wind_high"], label="High wind", color="g")
    axes[2].plot(df_slice["Date_Heure"], probs_df["wind_low"], label="Low wind", color="b")

    axes[3].plot(df_slice["Date_Heure"], df_slice["Wind_Norm"], label="Wind Norm")
    axes[3].plot(
        df_slice["Date_Heure"], 
        np.full(len(df_slice), 
        df_slice["Wind_Norm"].mean()), 
        label="Mean Wind Norm", color="r", linestyle="--")
    axes[3].plot(df_slice["Date_Heure"], df_slice["Wind_Norm"], label="wind norm")
    axes[3].plot(df_slice["Date_Heure"], df_slice["Wind_mean_3h"], label="wind mean 3h")
    axes[3].plot(df_slice["Date_Heure"], df_slice["Wind_Norm_lag_1h"], label="norm lag 1h")
    axes[3].plot(df_slice["Date_Heure"], df_slice["Wind_Norm_lag_24h"], label="norm lag 24h")

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("data/experts/experts_feat.csv")
    forecast = 100
    history = 6500 # 4500
    model = "BOA" # MLpol, MLprod, BOA, FTRL
    
    targets, experts, regime_features, valid_idx = prepare_features(df)

    idx_train = valid_idx[-history-forecast:-forecast]
    df_last24 = df.loc[valid_idx[-forecast:]]

    hmoe = train_hmoe(df, idx_train, model)

    y_pred_24h = rolling_forecast_24h(
        hmoe,
        df_last24,
        experts,
        regime_features,
    )

    # DEBUG
    print_expert_weights(hmoe, experts.columns)

    last_regime_feats = {
        k: v.loc[df_last24.index[-1]].values
        for k, v in regime_features.items()
    }
    print_regime_probs(hmoe, last_regime_feats)

    plot_24h_forecast(
        df_last24,
        df_last24[experts.columns],
        y_pred_24h,
        forecast
    )

    probs_df = extract_regime_probs(hmoe, df_last24, regime_features)
    # print(probs_df.describe())
    plot_regime_probs(probs_df, df_last24)
    


if __name__ == "__main__":
    main()