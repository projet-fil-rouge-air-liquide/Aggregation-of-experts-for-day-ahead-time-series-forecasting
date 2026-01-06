from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import numpy as np
import seaborn as sns

def wape(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Somme des erreurs absolues / Somme des valeurs réelles
    return (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100

def calculate_nmae(y_true, y_pred, capacity=None):
    """
    Calcule la nMAE. 
    Si capacity n'est pas fourni, on prend le max de y_true.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    
    if capacity is None:
        capacity = np.max(y_true) # Référence par rapport au pic historique
        
    return (mae / capacity) * 100

def fit_predict_eval(expert,X_train,X_test,y_train,y_test):
    expert.fit(X_train,y_train)
    y_pred = expert.predict(X_test)
    mae = (mean_absolute_error(y_test, y_pred))
    mse = (mean_squared_error(y_test, y_pred))
    wape_value = wape(y_test, y_pred)

    return y_pred,wape_value,mae,mse

def fit(expert,X_train,y_train):
    expert.fit(X_train,y_train)

def predict(expert,X_test):
    y_pred = expert.predict(X_test)
    return y_pred

def rmse(y_test,y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def to_24h_matrix(ser):
    # On s'assure d'avoir des journées complètes (multiple de 24)
    n = (len(ser) // 24) * 24
    return ser.values[:n].reshape(-1, 24)