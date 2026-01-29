# ============================================================
# APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE (NIGER, BURKINA, MALI)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestRegressor
import ee
import json
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(
    page_title="Surveillance Rougeole Multi-pays",
    layout="wide",
    page_icon="🦠",
)

st.title("🦠 Dashboard de Surveillance Prédictive – Rougeole")

# ============================================================
# 1. INITIALISATION GEE
# ============================================================

@st.cache_resource
def init_gee():
    try:
        key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key_dict["client_email"],
            key_data=json.dumps(key_dict)
        )
        ee.Initialize(credentials)
        return True
    except Exception as e:
        st.error("Erreur d’authentification Google Earth Engine")
        st.exception(e)
        return False

gee_ok = init_gee()
if not gee_ok:
    st.stop()

# ============================================================
# 2. SIDEBAR – DONNÉES ET PÉRIODE
# ============================================================

st.sidebar.header("📂 Données et période d'analyse")

# Pays
pays_selectionne = st.sidebar.selectbox("Sélectionner le pays", ["Niger", "Burkina Faso", "Mali"])

# Aires de santé
option_aire = st.sidebar.radio("Source Aires de Santé", ["GAUL Admin3 (GEE)", "Upload Shapefile/GeoJSON"])
if option_aire == "Upload Shapefile/GeoJSON":
    upload_file = st.sidebar.file_uploader("Charger un shapefile/GeoJSON", type=["shp", "geojson"])

# Linelist et vaccin
linelist_file = st.sidebar.file_uploader("Linelists rougeole (CSV)", type=["csv"])
vacc_file = st.sidebar.file_uploader("Couverture vaccinale (CSV – optionnel)", type=["csv"])

# Période
start_date = st.sidebar.date_input("Date de début", value=datetime(2024,1,1))
end_date = st.sidebar.date_input("Date de fin", value=datetime.today())

# ============================================================
# 3. CHARGEMENT AIRES DE SANTÉ
# ============================================================

@st.cache_resource
def load_gaul_admin3(pays):
    fc = ee.FeatureCollection("FAO/GAUL/2015/level3").filter(ee.Filter.eq("ADM0_NAME", pays))
    return fc

if option_aire == "GAUL Admin3 (GEE)":
    gaul_fc = load_gaul_admin3(pays_selectionne)
    Map = geemap.Map(center=[15,8], zoom=6)
    Map.addLayer(gaul_fc, {"color":"blue"}, "Aires de Santé")
    st.subheader("Carte interactive – Aires de Santé (GAUL)")
    st_folium(Map, width=900, height=650)
elif option_aire == "Upload Shapefile/GeoJSON":
    if upload_file:
        gdf = gpd.read_file(upload_file)
        Map = geemap.Map(center=[15,8], zoom=6)
        ee_fc = ee.FeatureCollection(gdf.__geo_interface__)
        Map.addLayer(ee_fc, {"color":"green"}, "Aires de Santé Uploadées")
        st.subheader("Carte interactive – Aires de Santé Uploadées")
        st_folium(Map, width=900, height=650)
    else:
        st.warning("Uploader un fichier pour continuer.")
        st.stop()

# ============================================================
# 4. LINELIST
# ============================================================

@st.cache_data
def generate_dummy_linelists(n=400):
    np.random.seed(42)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0,180,n), unit="D")
    return pd.DataFrame({
        "ID_Cas": range(1,n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(np.random.randint(1,5,n), unit="D"),
        "Aire_Sante": np.random.choice(["Niamey","Maradi","Zinder","Tahoua"], n),
        "Age_Mois": np.random.randint(6,180,n),
        "Statut_Vaccinal": np.random.choice(["Oui","Non"], n, p=[0.6,0.4])
    })

if linelist_file:
    df = pd.read_csv(linelist_file, parse_dates=["Date_Debut_Eruption","Date_Notification"])
else:
    st.info("Aucun linelist fourni – données simulées utilisées")
    df = generate_dummy_linelists()

# Filtre temporel
df = df[(df["Date_Debut_Eruption"] >= pd.to_datetime(start_date)) & 
        (df["Date_Debut_Eruption"] <= pd.to_datetime(end_date))]

# ============================================================
# 5. POPULATION – WORLDPOP (0-4 ans)
# ============================================================

@st.cache_data
def worldpop_children_stats(ee_fc):
    # M0-M4 + F0-F4
    bands = ["0","1","2","3","4"]
    pop = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic()
    pop_children = pop.select([f"M{b}" for b in bands]+[f"F{b}" for b in bands])
    stats = pop_children.reduceRegions(collection=ee_fc, reducer=ee.Reducer.sum(), scale=100)
    return stats

pop_fc = worldpop_children_stats(gaul_fc)

# ============================================================
# 6. URBANISATION – GHSL SMOD
# ============================================================

@st.cache_data
def urban_classification(fc):
    smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")
    def classify(feature):
        stats = smod.reduceRegion(ee.Reducer.mode(), feature.geometry(), scale=1000, maxPixels=1e9)
        return feature.set({"SMOD": stats.get("smod")})
    return fc.map(classify)

urban_fc = urban_classification(gaul_fc)

# ============================================================
# 7. CLIMAT – NASA POWER
# ============================================================

@st.cache_data(ttl=86400)
def fetch_climate_nasa_power(lat,lon,start_date,end_date):
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters":"T2M,PRECTOTCORR,RH2M",
        "community":"AG",
        "longitude":lon,
        "latitude":lat,
        "start":start_str,
        "end":end_str,
        "format":"JSON"
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200: return None
    data = r.json()
    if "properties" not in data: return None
    p = data["properties"]["parameter"]
    dates = list(p.get("RH2M", {}).keys())
    dfc = pd.DataFrame({
        "date": pd.to_datetime(dates,format="%Y%m%d"),
        "temp": [p.get("T2M",{}).get(d,np.nan) for d in dates],
        "precip": [p.get("PRECTOTCORR",{}).get(d,np.nan) for d in dates],
        "humidity": [p.get("RH2M",{}).get(d,np.nan) for d in dates]
    })
    return dfc

# ============================================================
# 8. PRÉPARATION TABLEAU DE BORD
# ============================================================

df["Delai_Notification"] = (df["Date_Notification"] - df["Date_Debut_Eruption"]).dt.days
df["Semaine"] = df["Date_Debut_Eruption"].dt.to_period("W").astype(str)

weekly = df.groupby(["Aire_Sante"]).agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x=="Non").mean()*100)
).reset_index()

st.subheader("📊 Tableau de bord – Indicateurs par Aire de Santé")
st.dataframe(weekly)

st.subheader("📈 Carte interactive – Cas Observés")
Map = geemap.Map(center=[15,8], zoom=6)
Map.addLayer(gaul_fc, {"color":"blue"}, "Aires de Santé")
st_folium(Map, width=900, height=650)

# ============================================================
# 9. PRÉDICTION – NOWCASTING / 12 SEMAINES
# ============================================================

weekly["Week_Index"] = range(len(weekly))
X = weekly[["Week_Index"]]
y = weekly["Cas_Observes"]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X,y)

next_weeks = np.array([[weekly["Week_Index"].max() + i] for i in range(1,13)])
pred_next = rf.predict(next_weeks)

st.subheader("🔮 Cas attendus sur 12 prochaines semaines")
st.line_chart(pred_next)

# ============================================================
# 10. ALERTES
# ============================================================

alert = weekly.copy()
alert["Alerte_Rouge"] = alert["Cas_Observes"] >= np.percentile(weekly["Cas_Observes"],75)
st.subheader("🚨 Aires de Santé en Alerte Rouge")
st.dataframe(alert[alert["Alerte_Rouge"]==True])
