from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns


def fit_predict_eval(expert,X_train,X_test,y_train,y_test):
    expert.fit(X_train,y_train)
    y_pred = expert.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return y_pred,rmse

def fit(expert,X_train,y_train):
    expert.fit(X_train,y_train)

def predict(expert,X_test):
    y_pred = expert.predict(X_test)
    return y_pred

def rmse(y_test,y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

