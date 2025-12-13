import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.Experts.expert_ElasticNet import ElasticNetExpert
from src.Experts.expert_LGBM import LGBMExpert
from src.Experts.expert_Ridge import RidgeExpert

############################
# Chargement des données   #
############################
Data_clean_Belgique = pd.read_csv("Data/Processed_data/data_engineering_belgique.csv")

N_BACKTEST = 2000
split_start = len(Data_clean_Belgique) - 24 - N_BACKTEST # on retire les 24 derniere donnée (pour etre prédite par moe plus tard)

results = pd.DataFrame()
results["Date_Heure"] = Data_clean_Belgique["Date_Heure"].iloc[split_start:split_start+N_BACKTEST].reset_index(drop=True)
results["y_true"] = Data_clean_Belgique["Eolien_MW"].iloc[split_start:split_start+N_BACKTEST].reset_index(drop=True)

############################################
# --- Fonction générique de backtesting --- #
############################################
def rolling_forecast(model, use_scaler=False):
    y_pred_list = []
    features = model.features
    if use_scaler:
        scaler = StandardScaler()
        scaler.fit(Data_clean_Belgique[features].iloc[:split_start])

    for i in range(N_BACKTEST):
        print(i)
        split_point = split_start + i
        train_df = Data_clean_Belgique.iloc[:split_point]

        X_train = train_df[features]
        y_train = train_df["Eolien_MW"]

        if use_scaler:
            X_train = scaler.transform(X_train)
            X_train = pd.DataFrame(X_train, columns=features)

        model.fit(X_train, y_train)

        X_next = Data_clean_Belgique[features].iloc[[split_point]]
        if use_scaler:
            X_next = scaler.transform(X_next)
            X_next = pd.DataFrame(X_next, columns=features)

        y_pred = model.predict(X_next)[0]
        y_pred_list.append(y_pred)

    return y_pred_list


#################
#     Ridge     #
#################
print("RIDGE")
re = RidgeExpert()
results["ridge"] = rolling_forecast(re, use_scaler=True)

####################
#       LGBM       #
####################
print("LGBM")
lgbm = LGBMExpert()
results["lgbm"] = rolling_forecast(lgbm, use_scaler=False)

######################
#     ElasticNet     #
######################
print("ElasticNet")
ene = ElasticNetExpert()
results["elasticnet"] = rolling_forecast(ene, use_scaler=True)

#save in csv 
results.to_csv("Data/Experts/experts.csv", index=False)

print("Saved in: Data/Experts/experts.csv")

###############################
# ---      métriques      --- #
###############################

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

metrics_table = pd.DataFrame(columns=["MAE", "RMSE", "MAPE (%)", "R2"], index=["ridge", "lgbm", "elasticnet"])

for model_name in ["ridge", "lgbm", "elasticnet"]:
    mae, rmse, mape, r2 = compute_metrics(results["y_true"], results[model_name])
    metrics_table.loc[model_name] = [mae, rmse, mape, r2]

metrics_table.to_csv("Data/Experts/experts_metrics.csv")
print("Saved in: Data/Experts/experts_metrics.csv")

print(metrics_table)
