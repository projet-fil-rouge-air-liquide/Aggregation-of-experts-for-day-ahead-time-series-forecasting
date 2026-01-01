import pandas as pd

# chargement des données
Data_clean_Belgique = pd.read_csv("Data/Processed_data/data_engineering_belgique.csv")
#df = Data_clean_Belgique.iloc[-n:].copy()
# définition du Label

y = Data_clean_Belgique["Eolien_MW"]

# features
X = Data_clean_Belgique.drop(columns="Eolien_MW")

# split des données et features et target - ATTENTION - ne pas mélanger passé et futur
n_Belgique = len(Data_clean_Belgique)

train_end = int(n_Belgique*(6/8))
valid_end = train_end + 2200

X_train = X.iloc[:train_end] # train servira à entrainer les experts
y_train = y.iloc[:train_end]

X_valid = X.iloc[train_end:valid_end]  # valid servira à entrainer les égrégateurs
y_valid = y.iloc[train_end:valid_end]  

X_test = X.iloc[valid_end:] # test servira à tester les experts et les agrégateurs
y_test = y.iloc[valid_end:]
