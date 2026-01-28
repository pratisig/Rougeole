# app.py
import streamlit as st
import geopandas as gpd
from gee_utils import init_gee, gdf_to_ee, worldpop_by_age, urban_ratio
from climate_utils import climate_polygon_mean
from risk_model import measles_risk, predict_cases
from viz_utils import risk_map

st.set_page_config("Measles Decision Support", layout="wide")

st.title("🧠 Measles Decision Support System")

init_gee()

geo = st.file_uploader("Upload aires de santé (GeoJSON)", type="geojson")
if geo:
    gdf = gpd.read_file(geo)

    ee_fc = gdf_to_ee(gdf)

    pop = worldpop_by_age(ee_fc)
    urban = urban_ratio(ee_fc)

    gdf["risk"] = gdf.apply(measles_risk, axis=1)

    st.metric("Aires à risque élevé", (gdf["risk"] > 70).sum())

    st_folium(risk_map(gdf), width=1200)
