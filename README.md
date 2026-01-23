Here is the English translation:

---

# Aggregation of Experts for Day-Ahead Time Series Forecasting

Project carried out as part of the **Capstone Project (Projet Fil Rouge)** of the **AI and Data Specialized Master’s Programs** at **Télécom Paris**.

This project aims to implement and compare several expert models for **day-ahead (J+1) time series forecasting**, and then aggregate them using a **Mixture of Experts (MOE)** approach.

---

## Contributors

* Alexandre Donnat
* Ambroise Laroye
* Héloïse Lordez
* Oscar De La Cruz
* William Jan

---

## General Project Structure

* `src/experts/`: construction and prediction of expert models
* `src/opera/`: implementation of the aggregation method (MOE)
* `src/data_cleaning.py`: data retrieval and cleaning
* `API_ERA5.py`: script for downloading meteorological data
* `data/`: storage of datasets (automatically generated)

---

## Data Loading

### 1. Meteorological Data (ERA5)

Meteorological data must be loaded **first**.
They require a personal account on the **Copernicus ERA5** platform.

#### Steps to follow:

1. Create an account at:
   [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Generate a **personal API key**
3. Run the data retrieval script:

   ```bash
   python API_ERA5.py
   ```

---

### 2. ELIA Data

```bash
python src/data_cleaning.py
```

No manual download is required.

---

## Repository Operation & Git Workflow

* Each contributor has read and write access to the repository.
* It is strongly recommended to:

  * Clone the `main` branch
  * Create a personal development branch (`dev/<firstname>` or equivalent)
  * Submit pull requests to `main` once features are validated

---

## Build Experts Dataset

### 1. Building Expert Models

```bash
python -m src.experts.build_experts
```

**Outputs:**

* `expert.csv`: expert predictions
* Comparison plot *Expert vs Ground Truth*

---

### 2. 24-Hour Predictions

```bash
python -m src.expertsprediction_for_24h
```

**Output:**

* `pred_24h.csv`: day-ahead (J+1) expert predictions

---

## Expert Aggregation (MOE)
Add regime features in experts.csv :
```bash
python -m src.opera.regime
```
Aggregate
```bash
python -m src.opera.hmoe
```

## Evaluate models

### Evaluation Protocol

The evaluation is performed on 10% of the dataset `data_engineering_belgique.csv`.

The test data is split into 24-hour windows (i.e., 24 consecutive values). To preserve and increase disparity of data and reduce computational cost. (because at each step we performed on 24 consecutive values - ie day ahead)

For each step:
* A value is observed.
* A prediction is made using **online learning over a 24-value horizon**, either based on the previous prediction or updated model parameters.
* A Mixture of Experts (MoE) is then applied to combine the experts’ predictions.
* The predictions from individual experts and from the MoE are compared to the true values to compute the error.

### Evaluation Metrics

The following metrics are used:

* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **R²** (Coefficient of Determination)

---

### Baseline: OPERA

To run the OPERA baseline evaluation:

```bash
python -m src.eval.moe
```

---

### HMOE (Regime bear/bull)

### Step 1: Add Market Regimes

Before running HMOE, market regimes (e.g., **bear / bull**) must be added to the dataset.

!! Make sure to update the dataset path if needed:
`data/processed_data/data_engineering_belgique.csv`

```bash
python -m src.opera.regime
```

### Step 2: Run HMOE Evaluation

```bash
python -m src.eval.hmoe
```

---

### Full Comparison

To evaluate and compare:

* Individual experts (**ElasticNet, Random Forest, LGBM**)
* OPERA baseline (MoE)
* HMOE with market regimes (bear/bull)

run:

```bash
python -m src.eval.moe_vs_hmoe_vs_experts
```

Output files save in csv in /eval

---
