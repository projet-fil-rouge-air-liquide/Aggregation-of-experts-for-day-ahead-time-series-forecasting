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
    nmae_value = calculate_nmae(y_test, y_pred)

    return y_pred,wape_value,mae,mse,nmae_value

def predict_eval(expert, X_test, y_test, capacity=None):
    # 1. Prédiction (renvoie probablement un vecteur plat de 2184 éléments)
    y_pred = expert.predict(X_test)
    
    # 2. On aplatit TOUT pour être sûr d'avoir des vecteurs (2184,)
    # y_test_flat aura 2184 éléments même s'il arrive en (91, 24)
    y_test_flat = np.array(y_test).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    
    # 3. Vérification de sécurité (optionnel mais recommandé)
    if y_test_flat.shape != y_pred_flat.shape:
        # Si X_test était au format 24h, y_pred pourrait être (91, 24)
        # On force le format plat pour la comparaison
        y_pred_flat = y_pred_flat.reshape(-1)

    # 4. Calcul des métriques sur les vecteurs plats
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    wape_value = (np.sum(np.abs(y_test_flat - y_pred_flat)) / np.sum(y_test_flat)) * 100
    
    if capacity is None:
        capacity = y_test_flat.max()
    nmae_value = (mae / capacity) * 100

    # On retourne y_pred au format original (91, 24) pour la suite de ton code
    return y_pred.reshape(-1, 24), wape_value, mae, mse, nmae_value

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