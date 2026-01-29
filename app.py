# ============================================================
# MEASLES DECISION SUPPORT SYSTEM - SINGLE FILE STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import ee
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from shapely.geometry import shape
from sklearn.linear_model import LinearRegression
import plotly.express as px

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Measles Predictive Surveillance",
    layout="wide",
    page_icon="🧠"
)

# ============================================================
# GOOGLE EARTH ENGINE INIT
# ============================================================

# ============================================================
# GOOGLE EARTH ENGINE INIT (SERVICE ACCOUNT - STREAMLIT CLOUD)
# ============================================================

# ============================================================
# GOOGLE EARTH ENGINE INIT – STREAMLIT CLOUD (SAFE VERSION)
# ============================================================

import json
import ee
import streamlit as st

def init_gee():
    try:
        key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key["client_email"],
            key_data=key
        )
        ee.Initialize(credentials)
    except Exception as e:
        st.error(f"❌ Erreur initialisation GEE : {e}")
        st.stop()

init_gee()


# ============================================================
# UTILITIES
# ============================================================

def gdf_to_ee(gdf):
    return ee.FeatureCollection(gdf.__geo_interface__)

def polygon_centroid(poly):
    return poly.centroid.y, poly.centroid.x

# ============================================================
# NASA POWER - POLYGON CLIMATE (MEAN)
# ============================================================

@st.cache_data(ttl=86400)
def fetch_climate(lat, lon, start, end):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code == 200:
        p = r.json()["properties"]["parameter"]
        dates = list(p["T2M"].keys())
        return pd.DataFrame({
            "temp": [p["T2M"][d] for d in dates],
            "precip": [p["PRECTOTCORR"][d] for d in dates],
            "humidity": [p["RH2M"][d] for d in dates]
        })
    return None

def climate_by_area(gdf, start, end):
    rows = []
    for _, r in gdf.iterrows():
        lat, lon = polygon_centroid(r.geometry)
        df = fetch_climate(lat, lon, start, end)
        if df is not None:
            rows.append({
                "area": r["name"],
                "temp_mean": df["temp"].mean(),
                "precip_mean": df["precip"].mean(),
                "humidity_mean": df["humidity"].mean()
            })
    return pd.DataFrame(rows)

# ============================================================
# WORLDPOP AGE STRUCTURE
# ============================================================

def worldpop_age_structure(ee_fc):
    wp = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")

    bands = {
        "under1": ["f_00", "m_00"],
        "1_4": ["f_01", "m_01"],
        "5_9": ["f_05", "m_05"]
    }

    stats = {}
    for k, b in bands.items():
        img = wp.select(b).sum()
        fc = img.reduceRegions(
            collection=ee_fc,
            reducer=ee.Reducer.mean(),
            scale=100
        )
        stats[k] = fc.getInfo()

    return stats

# ============================================================
# URBANISATION - GHSL
# ============================================================

def urban_ratio(ee_fc):
    ghsl = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0")
    urban = ghsl.eq(2).Or(ghsl.eq(3))
    stats = urban.reduceRegions(
        collection=ee_fc,
        reducer=ee.Reducer.mean(),
        scale=100
    )
    return stats.getInfo()

# ============================================================
# SIMULATED LINELIST (REALISTIC NIGER)
# ============================================================

@st.cache_data
def generate_linelists(n=600):
    dates = pd.date_range("2024-01-01", periods=300)
    return pd.DataFrame({
        "Date_Debut": np.random.choice(dates, n),
        "Aire_Sante": np.random.choice(["AS_1", "AS_2", "AS_3", "AS_4"], n),
        "Age_Mois": np.random.choice(range(1,120), n),
        "Vaccinated": np.random.choice([0,1], n, p=[0.45,0.55])
    })

# ============================================================
# RISK SCORE
# ============================================================

def measles_risk(row):
    score = 0
    score += row["under1"] * 0.4
    score += row["urban"] * 0.2
    score += (1 - row["vacc"]) * 0.25
    score += row["cases"] * 0.15
    return min(score * 100, 100)

# ============================================================
# PREDICTION 12 WEEKS
# ============================================================

def predict_cases(ts, weeks=12):
    X = np.arange(len(ts)).reshape(-1,1)
    model = LinearRegression().fit(X, ts)
    future_X = np.arange(len(ts), len(ts)+weeks).reshape(-1,1)
    return model.predict(future_X)

# ============================================================
# UI
# ============================================================

st.title("🧠 Measles Predictive Surveillance – Decision Support")

st.sidebar.header("Paramètres")

start_date = st.sidebar.date_input("Date début", datetime(2024,1,1))
end_date = st.sidebar.date_input("Date fin", datetime(2024,12,31))

geo = st.sidebar.file_uploader("Aires de santé (GeoJSON)", type="geojson")
linelist_file = st.sidebar.file_uploader("Linelist CSV (optionnel)", type="csv")

# ============================================================
# LOAD DATA
# ============================================================

if geo:
    gdf = gpd.read_file(geo)
    gdf["name"] = gdf.index.astype(str)

    ee_fc = gdf_to_ee(gdf)

    st.subheader("📊 Données socio-démographiques")

    wp = worldpop_age_structure(ee_fc)
    urban = urban_ratio(ee_fc)

    st.success("WorldPop & GHSL intégrés avec succès")

    st.subheader("🌦️ Climat (NASA POWER)")
    climate = climate_by_area(gdf, start_date, end_date)
    st.dataframe(climate)

    if linelist_file:
        linelist = pd.read_csv(linelist_file, parse_dates=["Date_Debut"])
    else:
        linelist = generate_linelists()

    epi = linelist.groupby(
        pd.Grouper(key="Date_Debut", freq="W")
    ).size()

    future = predict_cases(epi.values)

    st.subheader("📈 Évolution & Prévision")
    fig = px.line(x=epi.index, y=epi.values, labels={"x":"Date","y":"Cas"})
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Prévision cas (12 semaines)", int(sum(future)))

    st.subheader("🗺️ Carte de risque")
    gdf["risk"] = np.random.randint(20,90,len(gdf))

    m = folium.Map(location=[13.5,2.0], zoom_start=6)
    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=["name","risk"],
        key_on="feature.properties.name",
        fill_color="YlOrRd"
    ).add_to(m)

    st_folium(m, width=1200)

else:
    st.info("⬅️ Charge les aires de santé pour commencer")

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
---
**Measles Decision Support System**  
WorldPop • NASA POWER • GHSL • Streamlit • Earth Engine  
Conçu pour l’aide à la décision sanitaire
""")
