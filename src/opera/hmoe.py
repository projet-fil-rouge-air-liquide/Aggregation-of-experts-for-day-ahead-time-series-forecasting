import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.opera.mixture import HMoE
from src.opera.regime import Regime, SoftmaxGate, UpDownRegime, BullBearRegime, DayNightRegime, WindRegime, VolatilityRegime
from utils.hmoe_utils import *

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

def main():
    df = pd.read_csv("data/experts/predictions_experts_spe_feat.csv")
    forecast = 100
    history = 5000 # 6500 / 4500
    model = "BOA" # MLpol, MLprod, BOA, FTRL
    context = ("wind") # {} -> Opera baseline ; avaible regimes -> ("trend", "updown", "wind", "volatility", "daynight") 

    targets, experts, regime_features, valid_idx = prepare_features(df)

    idx_train = valid_idx[-history-forecast:-forecast]
    df_last24 = df.loc[valid_idx[-forecast:]]

    hmoe = train_hmoe(df, idx_train, model, context)

    # DEBUG
    # print_expert_weights(hmoe, experts.columns)

    y_pred_24h, weights_eff = rolling_forecast_online(
        hmoe,
        df_last24,
        experts,
        regime_features,
        targets
    )

    plot_24h_forecast(
        df_last24,
        df_last24[experts.columns],
        y_pred_24h,
        forecast
    )

    plot_effective_weights(weights_eff, df_last24)


if __name__ == "__main__":
    main()