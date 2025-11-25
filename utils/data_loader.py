import requests
import zipfile
import os
import pandas as pd

RTE_URLS = {
    "2021": "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_2021.zip",
   # "2022": "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_2022.zip",
   # "2023": "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_2023.zip",
}

def download_rte_zip(url: str, out_path: str):
    """
    Télécharge fichier ZIP depuis RTE et l'enregistrer localement.
    """
    print(f"Téléchargement depuis : {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()  # arrêt si erreur HTTP

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Fichier sauvegardé sous : {out_path}")

def extract_csv_from_zip(zip_path, output_dir="csv_output"):
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = [f for f in z.namelist() if f.endswith(".csv")][0]
        out_path = os.path.join(output_dir, csv_name)
        z.extract(csv_name, path=output_dir)
        print(f"CSV extrait dans : {out_path}")
        return out_path
    
def download_and_extract_all(output_zip="rte_zip", output_csv="csv_rte"):
    os.makedirs(output_zip, exist_ok=True)

    for year, url in RTE_URLS.items():
        zip_path = os.path.join(output_zip, f"RTE_{year}.zip")
        download_rte_zip(url, zip_path)
        extract_csv_from_zip(zip_path, output_csv)

if __name__ == "__main__":
    download_and_extract_all()


