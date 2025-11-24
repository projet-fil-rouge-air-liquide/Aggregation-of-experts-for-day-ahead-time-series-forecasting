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
PRODUCT = pd.read_csv("RAW_Data/registre-national-installation-production-stockage-electricite-agrege.csv",
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

                        # Extraction des puissances théoriques des parcs éoliens en Normandie
# sélectionner les parcs éoliens en Normandie pour les éoliennes en service, en Normandie, mis en service jusqu'au 31/122023
# Nettoyage des dates
PRODUCT["dateMiseEnS"] = pd.to_datetime(
    PRODUCT["dateMiseEnservice (format date)"], 
    errors="coerce"
)
# Filtrage éolien en Normandie
PRODUCT_filtered = PRODUCT[
    (PRODUCT["filiere"] == "Eolien") &
    (PRODUCT["regime"] == "En service") &
    (PRODUCT["region"] == "Normandie") &
    (PRODUCT["puisMaxInstallee"] > 0) &
    PRODUCT["dateMiseEnS"].notna()
].copy()

# convertir kW en MW
PRODUCT_filtered["MW_INST"] = PRODUCT_filtered["puisMaxInstallee"] / 1000

# Capacité installée par année
total_MW_INST_2021 = PRODUCT_filtered[PRODUCT_filtered["dateMiseEnS"] <= "2021-12-31"]["MW_INST"].sum()
total_MW_INST_2022 = PRODUCT_filtered[PRODUCT_filtered["dateMiseEnS"] <= "2022-12-31"]["MW_INST"].sum()
total_MW_INST_2023 = PRODUCT_filtered[PRODUCT_filtered["dateMiseEnS"] <= "2023-12-31"]["MW_INST"].sum()

# Timeline
MW_INST_timeline = pd.DataFrame({
    "Date_Heure": pd.date_range("2021-01-01", "2023-12-31 23:00:00", freq="H")
})

def get_MW_INST(date):
    if date <= pd.Timestamp("2021-12-31"):
        return total_MW_INST_2021
    elif date <= pd.Timestamp("2022-12-31"):
        return total_MW_INST_2022
    else:
        return total_MW_INST_2023
# application de la fonction
MW_INST_timeline["MW_INST"] = MW_INST_timeline["Date_Heure"].apply(get_MW_INST)
# sauvegarde de la timeline MW_INST
MW_INST_timeline.to_csv("Processed_data/MW_INST_timeline.csv", index=False)

                            # FUSION DATAFRAMES RTE/MTO/MW_INST
# ajout de la colonne MW_INST dans RTE
RTE = pd.merge(RTE, MW_INST_timeline, on="Date_Heure", how="left")
# fusion des deux dataframes sur la colonne Date_Heure et Perimetres
data = pd.merge(RTE, MTO, on=["Date_Heure"], how="inner")
# tri des données par Date_Heure
data = data.sort_values(by="Date_Heure").reset_index(drop=True)

                            # AJUSTEMENTS FINAUX
# modification des noms de colonnes
data = data.rename(columns={
    "u100":"speed_longitudinale_100m",
    "v100":"speed_latitudinale_100m",
    "msl":"mean_sea_level_pressure",
    "sp":"surface_pressure",
    "sst":"sea_surface_temperature",
    "t2m":"2m_temperature"
})

# ajout d'un index temporel
data["Date_Heure"] = pd.to_datetime(data["Date_Heure"])
data = data.set_index("Date_Heure",drop=False)

# sauvegarde des données nettoyées
data.to_csv("Processed_data/data_cleaned.csv", index=True)

                            # ANALYSE DES DONNES NETTOYEES  
# vérification des données manquantes
missing_data = data.isnull().sum()
print("\nDonnées manquantes par colonne :")
print(missing_data[missing_data > 0])       # pas de données manquantes
# vérifier les doublons
dups = data.index.duplicated().sum()
print("Nombre de doublons dans l’index :", dups)



