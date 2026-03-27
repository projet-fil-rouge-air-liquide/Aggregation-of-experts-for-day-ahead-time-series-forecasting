import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- DEBUG FUNCTIONS --- #
def print_expert_weights(hmoe, expert_names):
    for regime_tuple, mixture in hmoe.experts_by_regime.items():
        print(f"\nRegime {regime_tuple}")
        w = mixture.w
        for name, wi in zip(expert_names, w):
            print(f"{name:15s} | w = {wi:.4f}")


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

def rolling_forecast_online(hmoe, df_last24, experts, regime_features, targets):
    preds = []
    weights_history = []

    for idx in df_last24.index:

        # -------- PREDICT --------
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

        # -------- STORE WEIGHTS (avant update) --------
        weights_eff_t = compute_effective_weights(
            hmoe,
            df_last24.loc[[idx]],
            experts,
            regime_features
        )
        weights_history.append(weights_eff_t)

        # -------- ONLINE UPDATE --------
        y_true = targets.loc[idx]

        hmoe.update(
            expert_preds=expert_t,
            y_true=y_true,
            regime_features=regime_t,
        )

    weights_history = pd.concat(weights_history)

    return preds, weights_history

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
