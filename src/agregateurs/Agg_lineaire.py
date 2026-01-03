import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.colors import LogNorm

from src.config import FEATURES_EN, FEATURES_RIDGE, FEATURES_LGBM

from src.experts.expert_ElasticNet import ElasticNetExpert
from src.experts.expert_LGBM import LGBMExpert
from src.experts.expert_Ridge import RidgeExpert


# chargement des données
Data_clean_Belgique = pd.read_csv("Data/Processed_data/data_engineering_belgique.csv")
# définition du Label
labels_Belgique = Data_clean_Belgique["Eolien_MW"]

# split des données et features et target - ATTENTION - ne pas mélanger passé et futur
n_Belgique = len(Data_clean_Belgique)
train_end = int(n_Belgique*0.6)
valid_end = int(0.8*n_Belgique)

X_train = Data_clean_Belgique.iloc[:train_end]
y_train = labels_Belgique.iloc[:train_end]

X_valid = Data_clean_Belgique.iloc[train_end:valid_end]  # valid servira à comparer plusieurs modèle (utilisés après)
y_valid = labels_Belgique.iloc[train_end:valid_end]  # valid servira à comparer plusieurs modèle (utilisés après)

X_test = Data_clean_Belgique.iloc[valid_end:]
y_test = labels_Belgique.iloc[valid_end:]

# créer ExpertElasticNet
expert_EN = ElasticNetExpert(features=FEATURES_EN)
expert_LGBM = LGBMExpert(features=FEATURES_LGBM)
expert_Ridge = RidgeExpert(features=FEATURES_RIDGE)
experts = [expert_EN,expert_LGBM,expert_Ridge]
# fit/predict/evaluation
def fit_predict_eval(expert,X_train,X_test,y_train,y_test):
    expert.fit(X_train,y_train)
    y_pred = expert.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Expert - RMSE :", rmse)

    return y_pred,rmse

for exp in experts:
    fit_predict_eval(exp,X_train,X_test,y_train,y_test)


