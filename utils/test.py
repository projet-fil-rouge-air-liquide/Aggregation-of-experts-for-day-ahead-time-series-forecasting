import pandas as pd

data = pd.read_csv(
    "../Data/Raw_Data/ELIA.csv",
    sep=";",
    encoding="utf-8-sig",
    engine="python"
)

# trouver la dernière valeur de la colonne "Datetime"
print(data["Datetime"].iloc[-1])