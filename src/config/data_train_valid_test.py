import pandas as pd

# chargement des données
Data_clean_Belgique = pd.read_csv("Data/Processed_data/data_engineering_belgique_test.csv")
# définition du Label
labels_Belgique = Data_clean_Belgique["Eolien_MW"]

# split des données et features et target - ATTENTION - ne pas mélanger passé et futur
n_Belgique = len(Data_clean_Belgique)
train_end = int(n_Belgique*0.5)
valid_end = int(0.8*n_Belgique)

X_train = Data_clean_Belgique.iloc[:train_end] # train servira à entrainer les experts
y_train = labels_Belgique.iloc[:train_end]

X_valid = Data_clean_Belgique.iloc[train_end:valid_end]  # valid servira à entrainer les égrégateurs
y_valid = labels_Belgique.iloc[train_end:valid_end]  

X_test = Data_clean_Belgique.iloc[valid_end:] # test servira à tester les experts et les agrégateurs
y_test = labels_Belgique.iloc[valid_end:]