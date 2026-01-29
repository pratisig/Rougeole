# ============================================================
# APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE (NIGER, BURKINA, MALI)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
import ee
import json
import folium
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIG STREAMLIT
# ============================================================
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
pays_selectionne = st.sidebar.selectbox("Sélectionner le pays", ["Niger", "Burkina Faso", "Mali"])

# Aires de santé
option_aire = st.sidebar.radio("Source Aires de Santé", ["GAUL Admin3 (GEE)", "Upload Shapefile/GeoJSON"])
upload_file = None
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

@st.cache_data
def ee_fc_to_gdf(ee_fc):
    try:
        features = ee_fc.getInfo()["features"]
        gdf = gpd.GeoDataFrame.from_features(features)
        return gdf
    except Exception as e:
        st.error("Impossible de convertir FeatureCollection en GeoDataFrame")
        st.exception(e)
        return gpd.GeoDataFrame()

if option_aire == "GAUL Admin3 (GEE)":
    gaul_fc = load_gaul_admin3(pays_selectionne)
    sa_gdf = ee_fc_to_gdf(gaul_fc)
elif option_aire == "Upload Shapefile/GeoJSON":
    if upload_file:
        sa_gdf = gpd.read_file(upload_file)
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
        "Aire_Sante": np.random.choice(sa_gdf["ADM3_NAME"].unique(), n),
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
# Conversion en DataFrame
pop_gdf = ee_fc_to_gdf(pop_fc)
pop_df = pop_gdf[["ADM3_NAME","sum"]].rename(columns={"ADM3_NAME":"Aire_Sante","sum":"Pop_Totale"})

# ============================================================
# 6. URBANISATION – GHSL SMOD
# ============================================================
@st.cache_data
def urban_classification(ee_fc):
    smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")
    def classify(feature):
        stats = smod.reduceRegion(ee.Reducer.mode(), feature.geometry(), scale=1000, maxPixels=1e9)
        return feature.set({"SMOD": stats.get("smod")})
    return ee_fc.map(classify)

urban_fc = urban_classification(gaul_fc)
urban_gdf = ee_fc_to_gdf(urban_fc)
urban_df = urban_gdf[["ADM3_NAME","SMOD"]].rename(columns={"ADM3_NAME":"Aire_Sante","SMOD":"Urbanisation"})

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
# 8. PRÉPARATION FEATURES ET MODÉLISATION
# ============================================================
# Ajouter semaine
df["Semaine"] = df["Date_Debut_Eruption"].dt.to_period("W").astype(str)

weekly_features = df.groupby(["Aire_Sante","Semaine"]).agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x=="Non").mean()*100)
).reset_index()

weekly_features = weekly_features.merge(pop_df, on="Aire_Sante", how="left")
weekly_features = weekly_features.merge(urban_df, on="Aire_Sante", how="left")

# Coefficient climatique simulé pour test
weekly_features["Coef_Climatique"] = np.random.rand(len(weekly_features))*10

# Encoder variables
le_urban = LabelEncoder()
weekly_features["Urban_Encoded"] = le_urban.fit_transform(weekly_features["Urbanisation"].astype(str))

feature_cols = ["Cas_Observes","Non_Vaccines","Pop_Totale","Urban_Encoded","Coef_Climatique"]
X = weekly_features[feature_cols]
y = weekly_features["Cas_Observes"]

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X, y)

# ============================================================
# 9. PREDICTION 12 SEMAINES
# ============================================================
future_weeks = []
n_weeks = 12
latest_week_idx = len(weekly_features["Semaine"].unique())

for aire in weekly_features["Aire_Sante"].unique():
    aire_row = weekly_features[weekly_features["Aire_Sante"]==aire].iloc[-1]
    for i in range(1, n_weeks+1):
        future_weeks.append({
            "Aire_Sante": aire,
            "Semaine": f"Week_{latest_week_idx+i}",
            "Cas_Observes": aire_row["Cas_Observes"],
            "Non_Vaccines": aire_row["Non_Vaccines"],
            "Pop_Totale": aire_row["Pop_Totale"],
            "Urban_Encoded": aire_row["Urban_Encoded"],
            "Coef_Climatique": aire_row["Coef_Climatique"]
        })

future_df = pd.DataFrame(future_weeks)
future_df["Predicted_Cases"] = model.predict(future_df[feature_cols])

# ============================================================
# 10. RISQUE PAR AIRE
# ============================================================
risk_df = future_df.groupby("Aire_Sante").agg(
    Max_Predicted_Cases=("Predicted_Cases","max"),
    Week_of_Peak=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(),"Semaine"])
).reset_index()

# ============================================================
# 11. VISUALISATION CARTE
# ============================================================
# Fusion pour carte
sa_gdf = sa_gdf.merge(risk_df, left_on="ADM3_NAME", right_on="Aire_Sante", how="left")
max_cases = sa_gdf["Max_Predicted_Cases"].max()

colormap = folium.LinearColormap(["green","yellow","red"], vmin=0, vmax=max_cases, caption="Cas rouges prévus")
m = folium.Map(location=[15,8], zoom_start=6)
folium.GeoJson(
    sa_gdf,
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['Max_Predicted_Cases']) if feature['properties']['Max_Predicted_Cases'] is not None else "gray",
        'color':'black',
        'weight':1,
        'fillOpacity':0.7
    },
    tooltip=folium.GeoJsonTooltip(fields=["ADM3_NAME","Max_Predicted_Cases","Week_of_Peak"])
).add_to(m)
colormap.add_to(m)
st.subheader("🗺️ Carte des aires de santé – risque de rougeole")
st_folium(m, width=900, height=650)

# ============================================================
# 12. COURBES OBSERVÉES VS PRÉDITES
# ============================================================
st.subheader("📈 Courbes épidémiques – Observé vs Prévu")
plot_df = pd.concat([
    weekly_features[["Semaine","Cas_Observes","Aire_Sante"]],
    future_df.rename(columns={"Predicted_Cases":"Cas_Prevus"})[["Semaine","Cas_Prevus","Aire_Sante"]]
], axis=0)
fig1 = px.line(plot_df, x="Semaine", y="Cas_Observes", color="Aire_Sante", labels={"Cas_Observes":"Cas Observés"})
fig2 = px.line(plot_df, x="Semaine", y="Cas_Prevus", color="Aire_Sante", labels={"Cas_Prevus":"Cas Prévus"})
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 13. TABLEAUX INDICATEURS
# ============================================================
st.subheader("🚨 Aires de santé – risque maximal sur 12 semaines")
st.dataframe(risk_df.sort_values("Max_Predicted_Cases", ascending=False))

weekly_summary = df.groupby("Aire_Sante").agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x=="Non").mean()*100)
).reset_index()
st.subheader("📊 Tableau de bord – Indicateurs par Aire de Santé")
st.dataframe(weekly_summary)
