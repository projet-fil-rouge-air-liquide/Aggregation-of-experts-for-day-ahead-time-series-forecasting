Here is the English translation:

---

# 📊 Aggregation of Experts for Day-Ahead Time Series Forecasting

Project carried out as part of the **Capstone Project (Projet Fil Rouge)** of the **AI and Data Specialized Master’s Programs** at **Télécom Paris**.

This project aims to implement and compare several expert models for **day-ahead (J+1) time series forecasting**, and then aggregate them using a **Mixture of Experts (MOE)** approach.

---

## 👥 Contributors

* Alexandre Donnat
* Ambroise Laroye
* Héloïse Lordez
* Oscar De La Cruz
* William Jan

---

## 📁 General Project Structure

* `src/experts/`: construction and prediction of expert models
* `src/opera/`: implementation of the aggregation method (MOE)
* `src/data_cleaning.py`: data retrieval and cleaning
* `API_ERA5.py`: script for downloading meteorological data
* `data/`: storage of datasets (automatically generated)

---

## 📥 Data Loading

### 1. Meteorological Data (ERA5)

Meteorological data must be loaded **first**.
They require a personal account on the **Copernicus ERA5** platform.

#### Steps to follow:

1. Create an account at:
   👉 [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
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

👉 No manual download is required.

---

## 🔧 Repository Operation & Git Workflow

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

**Outputs:**

* Plot of **weights assigned to experts**
* Comparison **Experts vs MOE vs Ground Truth** over 24 hours

---
