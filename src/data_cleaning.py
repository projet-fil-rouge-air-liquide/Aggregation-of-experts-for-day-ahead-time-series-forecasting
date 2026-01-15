import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from utils.data_loader import download_and_extract_all

# # chargement des données
# download_and_extract_all()

# Donnnées Normandie
# MTO = pd.read_csv("../Data/Raw_Data/MTO_2021_22_23.csv",
#                   encoding="latin-1",
#                   sep=",")
# RTE_2021 = pd.read_csv("../Data/Raw_Data/RTE_Normandie_2021.csv",
#                   encoding="latin-1",
#                   sep=";")
# RTE_2022 = pd.read_csv("../Data/Raw_Data/RTE_Normandie_2022.csv",
#                   encoding="latin-1",
#                   sep=";")
# RTE_2023 = pd.read_csv("../Data/Raw_Data/RTE_Normandie_2023.csv",
#                   encoding="latin-1",
#                   sep=";")
# # concaténation des données RTE sur les 3 années
# RTE = pd.concat([RTE_2021, RTE_2022, RTE_2023], ignore_index=True)

# Données Belgique
ELIA = pd.read_csv("Data/Raw_Data/ELIA.csv",
    sep=";",
    encoding="utf-8-sig",
    engine="python")
MTO_Belgique = pd.read_csv("Data/Raw_Data/MTO_BELGIQUE_18_to_25.csv",
                  encoding="latin-1",
                  sep=",")

                            # NETTOYAGE DES DONNEES
# # sélection des colonnes utilies
# RTE = RTE[["Périmètre",
#            "Date",
#            "Heures",
#            "Eolien"]]

ELIA = ELIA[["Datetime",
             "Measured & Upscaled",
             "Offshore/onshore",
             "Region",
             ]]
# garder que les donnees offshore en Belgique et la région Federal
ELIA = ELIA[ELIA["Region"] == "Federal"]
ELIA = ELIA[ELIA["Offshore/onshore"] == "Offshore"]
# ne garder que la période 01/01/2018 au 01/12/2025
ELIA = ELIA[(ELIA["Datetime"] >= "2018-01-01") & (ELIA["Datetime"] <= "2025-12-01")]

# # renommage des colonnes
# RTE.columns = ["Perimetres",
#                "Date",
#                "Heure",
#                "Eolien_MW"]

ELIA = ELIA.rename(columns={
    "Datetime": "Date_Heure",
    "Measured & Upscaled": "Eolien_MW"
})


# conversion de colonne en DateTime
# RTE["Date"] = pd.to_datetime(RTE["Date"], format="%d/%m/%Y")
ELIA["Date_Heure"] = pd.to_datetime(ELIA["Date_Heure"], errors="coerce",utc=True)
ELIA["Date_Heure"] = ELIA["Date_Heure"].dt.tz_convert(None)

# création d'une colonne Date_Heure
# RTE["Date_Heure"] = RTE["Date"] + pd.to_timedelta(RTE["Heure"] + ":00")

# suppression des colonnes inutiles
# RTE = RTE.drop(columns=["Date", "Heure"])
ELIA = ELIA[["Date_Heure", "Eolien_MW", "Offshore/onshore", "Region"]]

                            # NETTOYAGE DES DONNEES MTO
# droper deux colonnes inutiles
# MTO = MTO.drop(columns=["latitude","longitude"],errors="ignore")
MTO_Belgique = MTO_Belgique.drop(columns=["latitude","longitude"],errors="ignore")
# conversion de la colonne Date en DateTime
# MTO["valid_time"] = pd.to_datetime(MTO["valid_time"], errors="coerce")
MTO_Belgique["valid_time"] = pd.to_datetime(MTO_Belgique["valid_time"], errors="coerce")

# création d'une colonne Date_Heure
# MTO = MTO.rename(columns={"valid_time": "Date_Heure"})
MTO_Belgique = MTO_Belgique.rename(columns={"valid_time": "Date_Heure"})

print("ELIA:", ELIA["Date_Heure"].dtype)
print("MTO_Belgique:", MTO_Belgique["Date_Heure"].dtype)
print(MTO_Belgique.head())

                            # FUSION DATAFRAMES RTE/MTO & ELIA/MTO
# fusion des deux dataframes sur la colonne Date_Heure et Perimetres
# data_Normandie = pd.merge(RTE, MTO, on=["Date_Heure"], how="inner")
data_Belgique = pd.merge(ELIA, MTO_Belgique, on=["Date_Heure"], how="inner")
# tri des données par Date_Heure
# data_Normandie = data_Normandie.sort_values(by="Date_Heure").reset_index(drop=True)
data_Belgique = data_Belgique.sort_values(by="Date_Heure").reset_index(drop=True)

                            # AJUSTEMENTS FINAUX
# # modification des noms de colonnes
# data_Normandie = data_Normandie.rename(columns={
#     "u100":"speed_longitudinale_100m",
#     "v100":"speed_latitudinale_100m",
#     "u10":"speed_longitudinale_10m",
#     "v10":"speed_latitudinale_10m",
#     "msl":"mean_sea_level_pressure",
#     "sp":"surface_pressure",
#     "sst":"sea_surface_temperature",
#     "t2m":"2m_temperature"
# })
data_Belgique = data_Belgique.rename(columns={
    "u100":"speed_longitudinale_100m",
    "v100":"speed_latitudinale_100m",
    "u10":"speed_longitudinale_10m",
    "v10":"speed_latitudinale_10m",
    "msl":"mean_sea_level_pressure",
    "sp":"surface_pressure",
    "sst":"sea_surface_temperature",
    "t2m":"2m_temperature"
})

# ajout d'un index temporel
# data_Normandie["Date_Heure"] = pd.to_datetime(data_Normandie["Date_Heure"])
# data_Normandie = data_Normandie.set_index("Date_Heure",drop=False)

data_Belgique["Date_Heure"] = pd.to_datetime(data_Belgique["Date_Heure"])
data_Belgique = data_Belgique.set_index("Date_Heure",drop=False)

# sauvegarde des données nettoyées
# data_Normandie.to_csv("../Data/Processed_data/data_cleaned_Normandie.csv", index=True)
data_Belgique.to_csv("Data/Processed_data/data_cleaned_Belgique.csv", index=True)
                            # ANALYSE DES DONNES NETTOYEES  
# vérification des données manquantes
data_Belgique = data_Belgique.dropna(subset=["Eolien_MW"])
# missing_data_Normandie = data_Normandie.isnull().sum()
missing_data_Belgique = data_Belgique.isnull().sum()

print("\nDonnées manquantes par colonne Belgique:")
print(missing_data_Belgique[missing_data_Belgique > 0])       # une donnée manquante en 2021



# print("\nDonnées manquantes par colonne Normandie:")
# print(missing_data_Normandie[missing_data_Normandie > 0])       # pas de données manquantes

# vérifier les doublons
# dups_Normandie = data_Normandie.index.duplicated().sum()
# print("Nombre de doublons dans l’index Normandie:", dups_Normandie)

dups_Belgique = data_Belgique.index.duplicated().sum()
print("Nombre de doublons dans l’index Belgique:", dups_Belgique)

