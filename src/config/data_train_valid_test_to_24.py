import numpy as np
import pandas as pd

from utils.fonction import to_24h_matrix
from src.config.data_train_valid_test import X_train,X_valid,X_test,y_train,y_valid,y_test


def to_24h_matrix(ser):
    # On s'assure d'avoir des journées complètes (multiple de 24)
    n = (len(ser) // 24) * 24
    return ser.values[:n].reshape(-1, 24)

# transformer les données heure par heure en journée (de 24h)
y_train_24 = to_24h_matrix(y_train)
y_valid_24 = to_24h_matrix(y_valid)
y_test_24  = to_24h_matrix(y_test)

X_train_24 = X_train.iloc[::24, :][:len(y_train_24)]
X_valid_24 = X_valid.iloc[::24, :][:len(y_valid_24)]
X_test_24  = X_test.iloc[::24, :][:len(y_test_24)]