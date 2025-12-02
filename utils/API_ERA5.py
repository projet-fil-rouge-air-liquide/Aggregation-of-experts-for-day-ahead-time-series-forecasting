import cdsapi

dataset = "reanalysis-era5-single-levels-timeseries"
request = {
    "variable": [
        "mean_sea_level_pressure",
        "surface_pressure",
        "sea_surface_temperature",
        "2m_temperature",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind"
    ],
    "location": {"longitude": 3, "latitude": 51.5},
    "date": ["2018-01-01/2025-12-01"],
    "data_format": "csv"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()