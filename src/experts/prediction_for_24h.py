import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.experts.expert_ElasticNet import ElasticNetExpert
from src.experts.expert_LGBM import LGBMExpert
from src.experts.expert_Ridge import RidgeExpert
from src.experts.expert_RandomForest import RandomForestExpert

Data_clean_Belgique = pd.read_csv("data/processed_data/data_engineering_belgique.csv")

n = len(Data_clean_Belgique) - 24 

results = pd.DataFrame()
results["Date_Heure"] = Data_clean_Belgique["Date_Heure"].iloc[n:].reset_index(drop=True)
results["y_true"] = Data_clean_Belgique["Eolien_MW"].iloc[n:].reset_index(drop=True)


def recursive_forecast(model, use_scaler=False):
    """
    Predict 24 hours: t+1, then t+2 using t+1 prediction, etc.
    """
    y_pred_list = []
    features = model.features

    # Prepare scaler
    if use_scaler:
        scaler = StandardScaler()
        scaler.fit(Data_clean_Belgique[features].iloc[:n])

    # Create a working copy of the dataset so we can inject predictions
    df = Data_clean_Belgique.copy()

    for i in range(24):
        print(f"Step {i+1}/24")

        split_point = n + i

        # Training set grows as predicted points are appended
        train_df = df.iloc[:split_point]

        X_train = train_df[features]
        y_train = train_df["Eolien_MW"]

        # Scale training data
        if use_scaler:
            X_train_scaled = scaler.transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, columns=features)

        # Train the model
        model.fit(X_train, y_train)

        # Build input for next prediction
        X_next = df[features].iloc[[split_point]]

        if use_scaler:
            X_next_scaled = scaler.transform(X_next)
            X_next = pd.DataFrame(X_next_scaled, columns=features)

        # Predict t+i+1
        y_pred = model.predict(X_next)[0]
        y_pred_list.append(y_pred)

        # Inject prediction so future steps use it
        df.loc[split_point, "Eolien_MW"] = y_pred

    return y_pred_list


# experts
print("\n=== RandomForest ===")
results["randomforest"] = recursive_forecast(
    RandomForestExpert(),
    use_scaler=False
)

print("\n=== LightGBM ===")
results["lgbm"] = recursive_forecast(
    LGBMExpert(),
    use_scaler=False
)

print("\n=== ElasticNet ===")
results["elasticnet"] = recursive_forecast(
    ElasticNetExpert(),
    use_scaler=True
)

#save in csv 
results.to_csv("data/experts/pred_24h.csv", index=False)

print("Saved in: data/experts/pred_24h.csv")
