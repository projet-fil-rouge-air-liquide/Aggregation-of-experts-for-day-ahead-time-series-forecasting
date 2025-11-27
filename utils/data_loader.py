import requests
import zipfile
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Data", "Raw_Data")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

TARGET_FILENAMES = {
    "2021": "RTE_Normandie_2021.csv",
    "2022": "RTE_Normandie_2022.csv",
    "2023": "RTE_Normandie_2023.csv"
}

def download_rte_zip(url: str):
    filename = url.split("/")[-1]
    out_path = os.path.join(RAW_DATA_DIR, filename)

    r = requests.get(url, stream=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return out_path


def extract_csv_from_zip(zip_path: str):

    year = zip_path.split("_")[-1].split(".")[0]

    with zipfile.ZipFile(zip_path, "r") as z:
        files = z.namelist()

        # Chercher CSV sinon XLSX
        candidates = [f for f in files if f.endswith(".csv") or f.endswith(".xlsx")]

        if not candidates:
            return None

        extracted_file = candidates[0]

        final_name = TARGET_FILENAMES.get(year, f"RTE_{year}.csv")
        final_path = os.path.join(RAW_DATA_DIR, final_name)

        z.extract(extracted_file, RAW_DATA_DIR)
        extracted_path = os.path.join(RAW_DATA_DIR, extracted_file)

        if extracted_file.endswith(".xlsx"):
            df = pd.read_excel(extracted_path)
            df.to_csv(final_path, index=False)
            os.remove(extracted_path)
        else:
            os.rename(extracted_path, final_path)

    os.remove(zip_path)

    return final_path


def data_already_loaded():
    return all(
        os.path.exists(os.path.join(RAW_DATA_DIR, fname))
        for fname in TARGET_FILENAMES.values()
    )


def download_and_extract_all():
    if data_already_loaded():
        return

    urls = [
        "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_2021.zip",
        "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_2022.zip",
        "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_2023.zip"
    ]

    for url in urls:
        zip_path = download_rte_zip(url)
        extract_csv_from_zip(zip_path)