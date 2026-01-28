# climate_utils.py
import requests
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

def polygon_centroid(polygon):
    return polygon.centroid.y, polygon.centroid.x

def climate_polygon_mean(gdf, start, end):
    rows = []

    for _, row in gdf.iterrows():
        lat, lon = polygon_centroid(row.geometry)

        df = fetch_climate_nasa_power(lat, lon, start, end)
        if df is not None:
            rows.append({
                "area": row["name"],
                "temp_mean": df["temp"].mean(),
                "precip_mean": df["precip"].mean()
            })

    return pd.DataFrame(rows)
