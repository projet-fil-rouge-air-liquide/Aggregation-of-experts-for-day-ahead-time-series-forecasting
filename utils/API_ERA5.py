import cdsapi

dataset = "reanalysis-era5-single-levels-timeseries"
request = {
    "variable": [
        "mean_sea_level_pressure",
        "surface_pressure",
        "sea_surface_temperature",
        "2m_temperature",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind"
    ],
    "location": {"longitude": 0, "latitude": 49.75},
    "date": ["2021-01-01/2023-12-31"],
    "data_format": "csv"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()