# =============================================================================
# DASHBOARD ROUGEOLE – CARTOGRAPHIE INTERACTIVE FOLIUM
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
import json
import ee
from shapely.geometry import mapping

st.set_page_config(page_title="Surveillance Rougeole", layout="wide", page_icon="🦠")
st.title("🦠 Dashboard de Surveillance Prédictive de la Rougeole")

# =============================================================================
# 1. INITIALISATION GOOGLE EARTH ENGINE
# =============================================================================
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
        st.error("Erreur GEE")
        st.exception(e)
        return False

gee_ok = init_gee()
if not gee_ok: st.stop()

# =============================================================================
# 2. SIDEBAR – CHOIX DES DONNÉES
# =============================================================================
st.sidebar.header("📂 Données & Période")
country = st.sidebar.selectbox("Pays", ["Niger", "Burkina Faso", "Mali"])
sa_option = st.sidebar.radio("Aires de Santé", ["GAUL Admin3 (GEE)", "Uploader shapefile / GeoJSON"])
sa_file = st.sidebar.file_uploader("Shapefile / GeoJSON", type=["shp","geojson"]) if sa_option=="Uploader shapefile / GeoJSON" else None
linelist_file = st.sidebar.file_uploader("Linelists CSV", type=["csv"])
use_dummy = st.sidebar.checkbox("Données simulées si CSV absent", True)
vacc_file = st.sidebar.file_uploader("Taux de vaccination (CSV – optionnel)", type=["csv"])
start_date = st.sidebar.date_input("Date début", datetime.today() - timedelta(days=180))
end_date = st.sidebar.date_input("Date fin", datetime.today())

# =============================================================================
# 3. AIRES DE SANTÉ
# =============================================================================
def load_gaul(country_name):
    code = {"Niger":"Niger","Burkina Faso":"Burkina Faso","Mali":"Mali"}
    fc = ee.FeatureCollection("FAO/GAUL/2015/level3").filter(ee.Filter.eq("ADM0_NAME", code[country_name]))
    return fc

def gdf_to_ee(gdf):
    features = [ee.Feature(mapping(f)) for f in gdf.geometry]
    return ee.FeatureCollection(features)

if sa_option=="GAUL Admin3 (GEE)":
    gaul_fc = load_gaul(country)
    sa_gdf = gpd.GeoDataFrame.from_features(ee.FeatureCollection(gaul_fc).getInfo()["features"])
else:
    if sa_file is not None:
        sa_gdf = gpd.read_file(sa_file)
        gaul_fc = gdf_to_ee(sa_gdf)
    else:
        st.warning("Aires manquantes – utilisation GAUL par défaut")
        gaul_fc = load_gaul(country)
        sa_gdf = gpd.GeoDataFrame.from_features(ee.FeatureCollection(gaul_fc).getInfo()["features"])

# =============================================================================
# 4. LINELISTS
# =============================================================================
def generate_dummy_linelists(n=400):
    np.random.seed(42)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0,180,n), unit="D")
    return pd.DataFrame({
        "ID_Cas": range(1,n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(np.random.randint(1,5,n), unit="D"),
        "Aire_Sante": np.random.choice(sa_gdf["ADM3_NAME"].tolist(), n),
        "Age_Mois": np.random.randint(0,120,n),
        "Statut_Vaccinal": np.random.choice(["Oui","Non"], n, p=[0.6,0.4])
    })

if linelist_file:
    df = pd.read_csv(linelist_file, parse_dates=["Date_Debut_Eruption","Date_Notification"])
elif use_dummy:
    st.info("Utilisation données simulées")
    df = generate_dummy_linelists()
else:
    st.error("Linelists requis")
    st.stop()

df = df[(df["Date_Debut_Eruption"]>=pd.to_datetime(start_date)) & (df["Date_Debut_Eruption"]<=pd.to_datetime(end_date))]

# =============================================================================
# 5. VACCINATION
# =============================================================================
if vacc_file:
    vacc = pd.read_csv(vacc_file)
else:
    vacc = pd.DataFrame({"Aire_Sante":sa_gdf["ADM3_NAME"].tolist(),"Couverture_Vaccinale":np.nan})

# =============================================================================
# 6. DONNÉES SOCIODEMOGRAPHIQUES – WorldPop enfants 0-9 ans
# =============================================================================
def worldpop_children_stats(fc):
    age_bands = ["0","1","2","3","4","5"]
    imgs = []
    for a in age_bands:
        imgs.append(ee.Image(f"WorldPop/GP/100m/pop_age_sex/{country.lower()}_m_{a}_2025_CN_100m_R2025A_v1"))
        imgs.append(ee.Image(f"WorldPop/GP/100m/pop_age_sex/{country.lower()}_f_{a}_2025_CN_100m_R2025A_v1"))
    pop_img = ee.ImageCollection(imgs).sum()
    stats = pop_img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100)
    return stats

pop_fc = worldpop_children_stats(gaul_fc).getInfo()
pop_df = pd.DataFrame([{"Aire_Sante":f["properties"]["ADM3_NAME"],"Pop_0_9":f["properties"]["sum"]} for f in pop_fc["features"]])

# =============================================================================
# 7. URBANISATION (GHSL SMOD)
# =============================================================================
def get_urban_class(fc):
    smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")
    def classify(feature):
        stats = smod.reduceRegion(ee.Reducer.mode(), feature.geometry(), scale=1000, maxPixels=1e9)
        return feature.set({"SMOD": stats.get("smod")})
    return fc.map(classify)

urban_fc = get_urban_class(gaul_fc).getInfo()
urban_df = pd.DataFrame([{"Aire_Sante":f["properties"]["ADM3_NAME"],
                          "Urbanisation":("Urbain" if f["properties"].get("SMOD",0)>=21 else "Rural" if f["properties"].get("SMOD",0)>=11 else "Non habité")} for f in urban_fc["features"]])

# =============================================================================
# 8. PRÉPARATION & FUSION
# =============================================================================
df["Delai_Notification"] = (df["Date_Notification"]-df["Date_Debut_Eruption"]).dt.days
df["Semaine"] = df["Date_Debut_Eruption"].dt.to_period("W").astype(str)

df = df.merge(pop_df, on="Aire_Sante", how="left")
df = df.merge(urban_df, on="Aire_Sante", how="left")
df = df.merge(vacc, on="Aire_Sante", how="left")

alert = df.groupby(["Aire_Sante","Semaine"]).agg(
    Cas=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x=="Non").mean()*100),
    Urbanisation=("Urbanisation","first")
).reset_index()

alert["Alerte_Rouge"] = (alert["Cas"]>=3) & (alert["Non_Vaccines"]>40)

# =============================================================================
# 9. PRÉDICTION
# =============================================================================
weekly = df.groupby(["Semaine"]).agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x=="Non").mean()*100)
).reset_index()

weekly["Week_Index"] = range(len(weekly))
X = weekly[["Week_Index","Cas_Observes","Non_Vaccines"]]
y = weekly["Cas_Observes"]

model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

future_weeks = 12
future_X = pd.DataFrame({
    "Week_Index": range(len(weekly), len(weekly)+future_weeks),
    "Cas_Observes": [weekly["Cas_Observes"].mean()]*future_weeks,
    "Non_Vaccines": [weekly["Non_Vaccines"].mean()]*future_weeks
})
future_preds = model.predict(future_X)

# =============================================================================
# 10. VISUALISATION FOLIUM
# =============================================================================
st.subheader("🌍 Carte interactive – Alerte Rouge")

m = folium.Map(location=[sa_gdf.geometry.centroid.y.mean(), sa_gdf.geometry.centroid.x.mean()], zoom_start=6, tiles="cartodbpositron")

def style_function(feature):
    aire = feature["properties"]["ADM3_NAME"]
    is_alert = alert[alert["Aire_Sante"]==aire]["Alerte_Rouge"].any()
    return {
        "fillOpacity":0.6,
        "weight":1,
        "color":"black",
        "fillColor":"red" if is_alert else "green"
    }

tooltip_fields = ["ADM3_NAME"]
tooltip_aliases = ["Aire de Santé : "]

folium.GeoJson(
    sa_gdf,
    style_function=style_function,
    tooltip=GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        localize=True
    )
).add_to(m)

st_folium(m, width=800, height=600)

# =============================================================================
# 11. KPI & GRAPHIQUES
# =============================================================================
st.subheader("📈 Épicourbe ajustée & prédiction 12 semaines")
import plotly.express as px
fig = px.line(weekly, x="Semaine", y="Cas_Observes", markers=True, labels={"Cas_Observes":"Cas observés"})
fig.add_scatter(x=[weekly["Semaine"].iloc[-1]+f" +{i}" for i in range(1,future_weeks+1)],
                y=future_preds, mode="lines+markers", name="Prévision")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🚨 Aires de Santé en Alerte Rouge")
st.dataframe(alert[alert["Alerte_Rouge"]==True])

st.metric("🔮 Cas attendus semaine prochaine", int(future_preds[0]))
