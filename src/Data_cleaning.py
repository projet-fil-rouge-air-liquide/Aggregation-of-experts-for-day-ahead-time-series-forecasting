# chargement des données
import pandas as pd
import numpy as np

MTO = pd.read_csv("RAW_data/MTO_2021_22_23.csv",
                  encoding="latin-1",
                  sep=",")
RTE_2021 = pd.read_csv("RAW_data/RTE_Normandie_2021.csv",
                  encoding="latin-1",
                  sep=";")
RTE_2022 = pd.read_csv("RAW_data/RTE_Normandie_2022.csv",
                  encoding="latin-1",
                  sep=";")
RTE_2023 = pd.read_csv("RAW_data/RTE_Normandie_2023.csv",
                  encoding="latin-1",
                  sep=";")

# concaténation des données RTE sur les 3 années
RTE = pd.concat([RTE_2021, RTE_2022, RTE_2023], ignore_index=True)

                            # NETTOYAGE DES DONNEES RTE
# sélection des colonnes utilies
RTE = RTE[["Périmètre",
           "Date",
           "Heures",
           "Eolien"]]
# renommage des colonnes
RTE.columns = ["Perimetres",
               "Date",
               "Heure",
               "Eolien_MW"]
# conversion de colonne en DateTime
RTE["Date"] = pd.to_datetime(RTE["Date"], format="%d/%m/%Y")
# création d'une colonne Date_Heure
RTE["Date_Heure"] = RTE["Date"] + pd.to_timedelta(RTE["Heure"] + ":00")
# suppression des colonnes inutiles
RTE = RTE.drop(columns=["Date", "Heure"])


                            # NETTOYAGE DES DONNEES MTO
# droper deux colonnes inutiles
MTO = MTO.drop(columns=["latitude","longitude"],errors="ignore")
# conversion de la colonne Date en DateTime
MTO["valid_time"] = pd.to_datetime(MTO["valid_time"], errors="coerce")
# création d'une colonne Date_Heure
MTO = MTO.rename(columns={"valid_time": "Date_Heure"})

                            # FUSION DATAFRAMES RTE ET MTO
# fusion des deux dataframes sur la colonne Date_Heure et Perimetres
data = pd.merge(RTE, MTO, on=["Date_Heure"], how="inner")
# tri des données par Date_Heure
data = data.sort_values(by="Date_Heure").reset_index(drop=True)
# sauvegarde des données nettoyées
data.to_csv("data_cleaned.csv", index=False)

# ajout d'un index temporel
data["Date_Heure"] = pd.to_datetime(data["Date_Heure"])
data = data.set_index("Date_Heure")


                            # ANALYSE DES DONNES NETTOYEES  
# vérification des données manquantes
missing_data = data.isnull().sum()
print("\nDonnées manquantes par colonne :")
print(missing_data[missing_data > 0])       # pas de données manquantes
# vérifier les doublons
dups = data.index.duplicated().sum()
print("Nombre de doublons dans l’index :", dups)

# vérification des valeurs absurbes 
print(data[data["Eolien_MW"] < 0])
print(data[data["Eolien_MW"] > 1500])
