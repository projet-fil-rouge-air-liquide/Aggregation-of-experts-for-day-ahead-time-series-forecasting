# Aggregation of Experts for Day-Ahead Time Series Forecasting

This project implements and compares several expert models for **day-ahead (J+1) time series forecasting**, and aggregates them using a **Mixture of Experts (MoE)** framework.

## Contributors

* Alexandre Donnat
* Ambroise Laroye
* Héloïse Lordez
* Oscar De La Cruz
* William Jan

## Project Structure

* `src/experts/`: expert model construction and prediction
* `src/opera/`: aggregation methods (MoE & HMoE)
* `src/eval/`: evaluation of experts, MoE, and HMoE
* `data/`: datasets

```bash
python src/data_cleaning.py
```

## Build Experts Dataset

### 1. Build Expert Models

```bash
python -m src.experts.build_experts
```

**Outputs:**

* `expert.csv`: expert predictions
* Comparison plot: *Experts vs Ground Truth*

Add regime features to `experts.csv`.
```bash
python -m src.experts.add_feature_regimes
```
**Output**: `experts_features.csv`
Expected columns:
```
Date_Heure|y_true|randomforest|lgbm|elasticnet|mom_24|mom_48|vol_12|vol_24|trend_strength|Wind_Norm|Wind_mean_3h|Wind_Norm_lag_1h|Wind_Norm_lag_24h
```

### 2. 24-Hour Predictions (Optional)

```bash
python -m src.experts.prediction_for_24h
```

**Output:**

* `pred_24h.csv`: day-ahead (J+1) expert predictions

---

## Expert Aggregation (MoE)

Ensure regime features are added to `experts.csv`.

```bash
python -m src.opera.moe
```

## Expert Aggregation (HMoE) – Trend (bear/bull) + Wind (high/low)

Set in `hmoe.py`:

* `model` (FTRL, BOA, …)
* `history`
* `forecast` horizon

```bash
python -m src.opera.hmoe
```

## Model Evaluation

### HMoE Evaluation

Evaluates only the **t+1 prediction** and compares it with experts.

Set:

* `model` (FTRL, BOA, …)
* `history`
* `test_step`: prediction and evaluation frequency

```bash
python -m src.eval.hmoe
```

**Metrics:**
MAE_mean, MAE_std, MAE_p90, MAE_p95, MAE_max, MAE_iqr

**Output:**
`eval_hmoe_vs_experts.csv`

