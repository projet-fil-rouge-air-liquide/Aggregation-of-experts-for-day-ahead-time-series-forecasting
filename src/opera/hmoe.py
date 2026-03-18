import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.opera.mixture import HMoE
from src.opera.regime import Regime, SoftmaxGate, UpDownRegime, BullBearRegime, DayNightRegime, WindRegime, VolatilityRegime

# DEBUG / ANALYSIS
def print_expert_weights(hmoe, expert_names):
    for regime_tuple, mixture in hmoe.experts_by_regime.items():
        print(f"\nRegime {regime_tuple}")
        w = mixture.w
        for name, wi in zip(expert_names, w):
            print(f"{name:15s} | w = {wi:.4f}")

def prepare_features(df):
    targets = df["y_true"]

    experts = df[[
        "LGBM_Global",
        "Ridge_Global",
        "RandomForest_Global",
        # "Ridge_Global_all_orient",
        # "RandomForest_Global_all_orient",
        # "LGBM_Global_all_orient",
        # "Ridge_Global_all_wind",
        # "RandomForest_Global_all_wind",
        # "LGBM_Global_all_wind",
        # "Ridge_Wind_orientation_NE",
        # "RandomForest_Wind_orientation_NE",
        # "LGBM_Wind_orientation_NE",
        # "Ridge_Wind_orientation_SE",
        # "RandomForest_Wind_orientation_SE",
        # "LGBM_Wind_orientation_SE",
        # "Ridge_Wind_orientation_NW",
        # "RandomForest_Wind_orientation_NW",
        # "LGBM_Wind_orientation_NW",
        # "Ridge_Wind_orientation_SW",
        # "RandomForest_Wind_orientation_SW",
        # "LGBM_Wind_orientation_SW",
        # "Ridge_Night",
        # "RandomForest_Night",
        # "LGBM_Night",
        # "Ridge_Day",
        # "RandomForest_Day",
        # "LGBM_Day",
        # "Ridge_Wind_Low",
        # "RandomForest_Wind_Low",
        # "LGBM_Wind_Low",
        # "Ridge_Wind_Med",
        # "RandomForest_Wind_Med",
        # "LGBM_Wind_Med",
        # "Ridge_Wind_High",
        # "RandomForest_Wind_High",
        # "LGBM_Wind_High",
        # "Ridge_Synoptique",
        # "RandomForest_Synoptique",
        # "LGBM_Synoptique",
        # "Ridge_Stationnar",
        # "RandomForest_Stationnar",
        # "LGBM_Stationnar"
        ]]

    df["hour"] = pd.to_datetime(df["Date_Heure"]).dt.hour

    regime_features = {
        "trend": df[[
            "trend_strength",
            "mom_24",
            "mom_48",
            "vol_24",
        ]],
        "wind": df[[
            "Wind_Norm",
            # "Wind_mean_3h",
            # "Wind_Norm_lag_1h", 
            # "Wind_Norm_lag_24h"
        ]],
        "daynight": df[[
            "hour"
        ]]
    }
    regime_features["updown"] = regime_features["trend"]
    regime_features["volatility"] = regime_features["trend"]

    valid_idx = (
        targets.dropna().index
        .intersection(experts.dropna().index)
        .intersection(regime_features["trend"].dropna().index)
        .intersection(regime_features["wind"].dropna().index)
        .intersection(regime_features["daynight"].dropna().index)
    )

    return targets, experts, regime_features, valid_idx

def train_hmoe(df, idx_train, model, context=("trend", "wind")):
    targets, experts, regime_features, _ = prepare_features(df)
    y_train = targets.loc[idx_train]
    X_train = experts.loc[idx_train]
    regime_train = {k: v.loc[idx_train] for k, v in regime_features.items()}
    regime_context = {}

    if "trend" in context:
        trend_regime = Regime(
            name="trend",
            regimes=["bull", "bear"],
            predictor=SoftmaxGate(2),
            prior=BullBearRegime(),
        )
        regime_context["trend"] = trend_regime

    if "updown" in context:
        updown_regime = Regime(
            name="updown",
            regimes=["up", "down"],
            predictor=SoftmaxGate(2),
            prior=UpDownRegime(trend_idx=0),
        )
        regime_context["updown"] = updown_regime

    if "volatility" in context:
        vol_idx = regime_features["volatility"].columns.get_loc("vol_24")
        vol_series = regime_features["volatility"].iloc[idx_train, vol_idx].values
        low_th  = np.quantile(vol_series, 0.3)
        high_th = np.quantile(vol_series, 0.7)
        volatility_regime = Regime(
            name="volatility",
            regimes=["low_vol", "high_vol"],
            predictor=SoftmaxGate(2),
            prior=VolatilityRegime(
                vol_idx=vol_idx,
                low_th=low_th,
                high_th=high_th,
                strength=0.3
            ),
        )
        regime_context["volatility"] = volatility_regime

    if "wind" in context:
        wind_std = np.std(regime_features["wind"].iloc[idx_train, 0].values)
        wind_mean = np.mean(regime_features["wind"].iloc[idx_train, 0].values)
        wind_regime = Regime(
            name="wind",
            regimes=["low", "high"],
            predictor=WindRegime(
                wind_feature_idx=0,
                wind_mean=wind_mean,
                wind_std=wind_std,
                strength=2.0
            ),
            prior=None
        )
        regime_context["wind"] = wind_regime

    if "daynight" in context:
        daynight_regime = Regime(
            name="daynight",
            regimes=["day", "night"],
            predictor=DayNightRegime(hour_idx=0),
            prior=None
        )
        regime_context["daynight"] = daynight_regime


    if model=="FTRL":
        hmoe = HMoE(
            y=y_train,
            experts=X_train,
            regime_context=regime_context,
            model="FTRL",
            loss_type="mse",
            parameters={"eta": 0.01, "l1": 0.0, "l2": 0.01},
        )
    elif model in {"BOA", "MLprod", "MLpol"}:
        hmoe = HMoE(
            y=y_train,
            experts=X_train,
            regime_context=regime_context,
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

    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def compute_effective_weights(hmoe, df_slice, experts, regime_features):

    rows = []

    for idx in df_slice.index:

        regime_probs = {}

        for name, context in hmoe.regime_context.items():
            p = context.predict(regime_features[name].loc[idx].values)
            regime_probs[name] = dict(zip(context.regimes, p))

        row = {"index": idx}

        for expert_i, expert_name in enumerate(experts.columns):

            w_eff = 0

            for regime_tuple, mixture in hmoe.experts_by_regime.items():

                prob = 1
                for regime_name, regime_value in zip(hmoe.regime_context.keys(), regime_tuple):
                    prob *= regime_probs[regime_name][regime_value]

                w_eff += prob * mixture.w[expert_i]

            row[expert_name] = w_eff

        rows.append(row)

    return pd.DataFrame(rows).set_index("index")

def plot_effective_weights(weights_eff, df_slice):

    plt.figure(figsize=(14,6))

    for col in weights_eff.columns:
        plt.plot(
            df_slice["Date_Heure"],
            weights_eff[col],
            label=col,
            lw=2
        )

    plt.title("Effective expert weights")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("data/experts/predictions_experts_spe_feat.csv")
    forecast = 100
    history = 5000 # 6500 / 4500
    model = "BOA" # MLpol, MLprod, BOA, FTRL
    context = ("trend", "updown", "volatility") # {} -> Opera baseline ; ("trend", "updown", "wind", "volatility", "daynight") 

    targets, experts, regime_features, valid_idx = prepare_features(df)

    idx_train = valid_idx[-history-forecast:-forecast]
    df_last24 = df.loc[valid_idx[-forecast:]]

    hmoe = train_hmoe(df, idx_train, model, context)

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

    plot_24h_forecast(
        df_last24,
        df_last24[experts.columns],
        y_pred_24h,
        forecast
    )

    probs_df = extract_regime_probs(hmoe, df_last24, regime_features)

    # plot_regime_probs(probs_df, df_last24)

    weights_eff = compute_effective_weights(
        hmoe,
        df_last24,
        experts,
        regime_features
    )

    plot_effective_weights(weights_eff, df_last24)


if __name__ == "__main__":
    main()