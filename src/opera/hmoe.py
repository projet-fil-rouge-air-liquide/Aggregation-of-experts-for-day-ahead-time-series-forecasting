import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.opera.mixture import HierarchicalHorizonOPERA, RegimeGate


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


# ---------------------------------------------------------------------
# MODEL INIT
# ---------------------------------------------------------------------

def initialize_opera_model(
    targets: pd.Series,
    experts: pd.DataFrame,
    regimes: list[str],
    horizons: list[int],
    parameters: dict
) -> HierarchicalHorizonOPERA:
    return HierarchicalHorizonOPERA(
        y=targets.values,
        experts=experts,
        regimes=regimes,
        horizons=horizons,
        regime_gate=RegimeGate(n_regimes=len(regimes), lr=0.1),
        model="FTRL",
        loss_type="mse",
        parameters=parameters,
    )


# ---------------------------------------------------------------------
# ONLINE LEARNING
# ---------------------------------------------------------------------

def online_learning(
    hmod: HierarchicalHorizonOPERA,
    experts: pd.DataFrame,
    targets: pd.Series,
    regime_features: pd.DataFrame,
    horizons: list[int]
) -> None:
    N_train = len(targets) - max(horizons)

    for t in tqdm(range(N_train), desc="OPERA online learning"):
        expert_preds_t = {h: experts.iloc[[t]] for h in horizons}
        y_true_t = {h: targets.iloc[t + h] for h in horizons}

        hmod.update(
            expert_preds=expert_preds_t,
            y_true=y_true_t,
            regime_features=regime_features.iloc[t].values,
            regime_label=None  # ignoré
        )


# ---------------------------------------------------------------------
# DEBUG / ANALYSIS
# ---------------------------------------------------------------------

def debug_regime_gate(hmod, regime_features):
    x = regime_features.iloc[-1].values
    p = hmod.regime_gate.predict(x)

    print("\n=== REGIME GATE ===")
    for i, r in enumerate(hmod.regimes):
        print(f"P({r}) = {p[i]:.3f}")
    print("Dominant regime:", hmod.regimes[np.argmax(p)])


def print_opera_weights(hmod, p_regime, expert_names):
    """Print OPERA weights by regime and horizon."""
    print("\n=== OPERA WEIGHTS (internal, by regime and horizon) ===")

    for r in hmod.regimes:
        print(f"\nRegime: {r}")
        for h in [1, 8, 16, 24]:
            if h in hmod.opera[r]:
                w = hmod.opera[r][h].w
                print(f"  h={h:2d} | w={np.round(w, 3)}")

    # --- effective weights ---
    w_eff = np.zeros(len(expert_names))
    for i, r in enumerate(hmod.regimes):
        w_r = hmod.opera[r][max(hmod.horizons)].w
        w_eff += p_regime[i] * w_r

    print("\n=== EFFECTIVE WEIGHTS PER EXPERT ===")
    for name, w in zip(expert_names, w_eff):
        print(f"{name:15s} | w_eff = {w:.3f}")



# ---------------------------------------------------------------------
# PREDICTION + PLOT
# ---------------------------------------------------------------------

def predict_and_plot(
    hmod,
    df_last24,
    experts_last24,
    regime_features,
    horizons
):
    h_max = max(horizons)

    plt.figure(figsize=(15, 7))

    # --- Ground truth ---
    plt.plot(
        df_last24["Date_Heure"],
        df_last24["y_true"],
        label="y_true",
        lw=2.5
    )

    # --- Experts ---
    for col in experts_last24.columns:
        plt.plot(
            df_last24["Date_Heure"],
            df_last24[col],
            "--",
            alpha=0.6,
            label=col
        )

    # --- Rolling OPERA prediction ---
    y_curve = []

    for t in range(len(df_last24)):
        x_gate_t = regime_features.iloc[-len(df_last24) + t].values

        expert_preds_t = {
            h: experts_last24.iloc[[t]]
            for h in horizons
        }

        y_pred_t = hmod.predict(expert_preds_t, x_gate_t)
        y_curve.append(y_pred_t[h_max].item())


    plt.plot(
        df_last24["Date_Heure"],
        y_curve,
        lw=2.5,
        label=f"Hierarchical OPERA ({h_max}h)"
    )

    plt.title("24h Prediction – OPERA (rolling forecast)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Regime mix at final time ---
    p_regime = hmod.regime_gate.predict(
        regime_features.iloc[-1].values
    )

    print("\n=== REGIME MIX (final step) ===")
    for i, r in enumerate(hmod.regimes):
        print(f"{r:4s} | p = {p_regime[i]:.3f}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    df = load_data("data/experts/experts.csv")

    targets = df["y_true"]
    experts = df[["randomforest", "lgbm", "elasticnet"]]
    regime_features = df[["ret_1", "ret_24", "vol_24", "mom_24", "hour_sin", "hour_cos"]]

    # dropna first row invalid
    valid_idx = regime_features.dropna().index
    df = df.loc[valid_idx]
    targets = targets.loc[valid_idx]
    experts = experts.loc[valid_idx]
    regime_features = regime_features.loc[valid_idx]


    horizons = [1, 8, 16, 24]
    regimes = ["bull", "bear"]

    parameters = {"eta": 0.05, "l1": 0.0, "l2": 0.01}

    hmod = initialize_opera_model(
        targets, experts, regimes, horizons, parameters
    )

    online_learning(
        hmod, experts, targets, regime_features, horizons
    )
    print("\n=== REGIME GATE WEIGHTS ===")
    print(hmod.regime_gate.W)

    debug_regime_gate(hmod, regime_features)

    df_last24 = df.tail(24)
    experts_last24 = df_last24[experts.columns]

    predict_and_plot(
        hmod,
        df_last24,
        experts_last24,
        regime_features,
        horizons
    )

    p_regime = hmod.regime_gate.predict(regime_features.iloc[-1].values)
    print_opera_weights(hmod, p_regime, experts.columns)


if __name__ == "__main__":
    main()
