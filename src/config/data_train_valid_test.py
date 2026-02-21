import pandas as pd

# chargement des données
Data_clean_Belgique = pd.read_csv("Data/Processed_data/data_engineering_belgique.csv")
Data_clean_Belgique['Date_Heure'] = pd.to_datetime(Data_clean_Belgique['Date_Heure'])

# initialisation au premier minuit avec comme premier jour 2023-01-01 00:00:00 pour limiter le temps de calcul à deux ans
first_midnight = Data_clean_Belgique[Data_clean_Belgique['Date_Heure'] == '2022-01-01 00:00:00'].index[0]
df = Data_clean_Belgique.iloc[first_midnight:].reset_index(drop=True)

# split des données et features et target
n_total = (len(df) // 24) * 24
df = df.iloc[:n_total]

# définition du Label et features
y = df["Eolien_MW"]
X = df.drop(columns="Eolien_MW")
# bornes train/valid
duree_test = 2184   # 3 mois (91 jours)
duree_valid = 8640  # 6 mois (2 saisons)
# On calcule les points de coupure en partant de la FIN (n_total)
valid_end = n_total - duree_test
train_end = valid_end - duree_valid

# création des données
X_train, y_train = X.iloc[:train_end], y.iloc[:train_end] # train servira à entrainer les experts
X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]  # valid servira à entrainer les égrégateurs
X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:] # test servira à tester les experts et les agrégateurs


print(f"Début Train : {X_train['Date_Heure'].iloc[0]}  | Fin Train : {X_train['Date_Heure'].iloc[-1]}")
print(f"Début Valid : {X_valid['Date_Heure'].iloc[0]}  | Fin Valid : {X_valid['Date_Heure'].iloc[-1]}")
print(f"Début Test  : {X_test['Date_Heure'].iloc[0]}   | Fin Test  : {X_test['Date_Heure'].iloc[-1]}")