# ============================================================
# APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE (Multi-pays)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import ee
import json
import folium
import geemap
from streamlit_folium import st_folium

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

# Pays
pays_selectionne = st.sidebar.selectbox("Sélectionner le pays", ["Niger", "Burkina Faso", "Mali"])

# Aires de santé
option_aire = st.sidebar.radio("Source Aires de Santé", ["GAUL Admin3 (GEE)", "Upload Shapefile/GeoJSON"])
upload_file = None
if option_aire == "Upload Shapefile/GeoJSON":
    upload_file = st.sidebar.file_uploader("Charger un shapefile/GeoJSON", type=["shp","geojson"])

# Linelist et vaccin
linelist_file = st.sidebar.file_uploader("Linelists rougeole (CSV)", type=["csv"])
vacc_file = st.sidebar.file_uploader("Couverture vaccinale (CSV – optionnel)", type=["csv"])

# Période
start_date = st.sidebar.date_input("Date de début", value=datetime(2024,1,1))
end_date = st.sidebar.date_input("Date de fin", value=datetime.today())

# ============================================================
# 3. CHARGEMENT AIRES DE SANTÉ
# ============================================================
@st.cache_data
def ee_fc_to_gdf_safe(fc):
    """Convertir FeatureCollection GEE en GeoDataFrame sans getInfo() direct"""
    try:
        gdf = geemap.ee_to_geopandas(fc)
        if "ADM3_NAME" in gdf.columns:
            gdf["ADM3_NAME"] = gdf["ADM3_NAME"].astype(str)
        return gdf
    except Exception as e:
        st.error("Impossible de récupérer les données GAUL Admin3 via GEE")
        st.exception(e)
        return gpd.GeoDataFrame()

# Charger GAUL ou fichier uploadé
if option_aire == "GAUL Admin3 (GEE)":
    fc = ee.FeatureCollection("FAO/GAUL/2015/level3") \
           .filter(ee.Filter.eq("ADM0_NAME",pays_selectionne))
    sa_gdf = ee_fc_to_gdf_safe(fc)
    if sa_gdf.empty:
        st.warning("GAUL Admin3 vide, passer à l’upload de shapefile")
    else:
        st.success(f"{len(sa_gdf)} aires de santé GAUL chargées")
elif option_aire == "Upload Shapefile/GeoJSON":
    if upload_file:
        sa_gdf = gpd.read_file(upload_file)
        # Vérification colonne
        if "ADM3_NAME" not in sa_gdf.columns:
            sa_gdf["ADM3_NAME"] = sa_gdf.columns[0]  # fallback si pas de col ADM3_NAME
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
def worldpop_children_stats(sa_gdf):
    try:
        fc = geemap.geopandas_to_ee(sa_gdf)
        bands = ["0","1","2","3","4"]
        pop = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic()
        pop_children = pop.select([f"M{b}" for b in bands]+[f"F{b}" for b in bands])
        stats = pop_children.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100)
        # Convertir en GeoDataFrame
        gdf = geemap.ee_to_geopandas(stats)
        gdf["Pop_Totale"] = gdf[[f"M{b}" for b in bands]+[f"F{b}" for b in bands]].sum(axis=1)
        return gdf[["ADM3_NAME","Pop_Totale"]]
    except:
        return pd.DataFrame({"ADM3_NAME": sa_gdf["ADM3_NAME"], "Pop_Totale": np.random.randint(1000,5000,len(sa_gdf))})

pop_df = worldpop_children_stats(sa_gdf)

# ============================================================
# 6. URBANISATION – GHSL SMOD
# ============================================================
@st.cache_data
def urban_classification(sa_gdf):
    fc = geemap.geopandas_to_ee(sa_gdf)
    smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")
    def classify(feature):
        stats = smod.reduceRegion(ee.Reducer.mode(), feature.geometry(), scale=1000, maxPixels=1e9)
        return feature.set({"SMOD": stats.get("smod")})
    urban_fc = fc.map(classify)
    urban_gdf = geemap.ee_to_geopandas(urban_fc)
    urban_gdf.rename(columns={"SMOD":"Urbanisation"}, inplace=True)
    return urban_gdf[["ADM3_NAME","Urbanisation"]]

urban_df = urban_classification(sa_gdf)

# ============================================================
# 7. CLIMAT – NASA POWER (exemple sur centroid de l’aire)
# ============================================================
@st.cache_data(ttl=86400)
def fetch_climate_nasa_power(sa_gdf, start_date, end_date):
    data_list = []
    for idx,row in sa_gdf.iterrows():
        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters":"T2M,PRECTOTCORR,RH2M",
            "community":"AG",
            "longitude":lon,
            "latitude":lat,
            "start":start_date.strftime("%Y%m%d"),
            "end":end_date.strftime("%Y%m%d"),
            "format":"JSON"
        }
        try:
            r = requests.get(url, params=params, timeout=60)
            j = r.json()
            if "properties" in j:
                p = j["properties"]["parameter"]
                coef = np.nanmean([p.get("T2M",{}).get(d,0) for d in p.get("T2M",{})])
                data_list.append({"ADM3_NAME":row["ADM3_NAME"],"Coef_Climatique":coef})
        except:
            data_list.append({"ADM3_NAME":row["ADM3_NAME"],"Coef_Climatique":np.nan})
    return pd.DataFrame(data_list)

climate_df = fetch_climate_nasa_power(sa_gdf, start_date, end_date)

# ============================================================
# 8. Préparer features et prédiction rougeole
# ============================================================
df["Semaine"] = df["Date_Debut_Eruption"].dt.to_period("W").astype(str)

weekly_features = df.groupby(["Aire_Sante","Semaine"]).agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x:(x=="Non").mean()*100)
).reset_index()

# Fusion population, urban, climat
weekly_features = weekly_features.merge(pop_df.rename(columns={"ADM3_NAME":"Aire_Sante"}), on="Aire_Sante", how="left")
weekly_features = weekly_features.merge(urban_df.rename(columns={"ADM3_NAME":"Aire_Sante"}), on="Aire_Sante", how="left")
weekly_features = weekly_features.merge(climate_df.rename(columns={"ADM3_NAME":"Aire_Sante"}), on="Aire_Sante", how="left")

# Encoder Urbanisation
le_urban = LabelEncoder()
weekly_features["Urban_Encoded"] = le_urban.fit_transform(weekly_features["Urbanisation"].astype(str))

feature_cols = ["Cas_Observes","Non_Vaccines","Pop_Totale","Urban_Encoded","Coef_Climatique"]
X = weekly_features[feature_cols]
y = weekly_features["Cas_Observes"]

# Modèle prédictif
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X,y)

# Prévisions 12 semaines
future_weeks = []
n_weeks = 12
latest_week_idx = len(weekly_features["Semaine"].unique())
for aire in weekly_features["Aire_Sante"].unique():
    aire_row = weekly_features[weekly_features["Aire_Sante"]==aire].iloc[-1]
    for i in range(1,n_weeks+1):
        future_weeks.append({
            "Aire_Sante":aire,
            "Semaine":f"Week_{latest_week_idx+i}",
            "Cas_Observes":aire_row["Cas_Observes"],
            "Non_Vaccines":aire_row["Non_Vaccines"],
            "Pop_Totale":aire_row["Pop_Totale"],
            "Urban_Encoded":aire_row["Urban_Encoded"],
            "Coef_Climatique":aire_row["Coef_Climatique"]
        })
future_df = pd.DataFrame(future_weeks)
future_df["Predicted_Cases"] = model.predict(future_df[feature_cols])

# Classement risque
risk_df = future_df.groupby("Aire_Sante").agg(
    Max_Predicted_Cases=("Predicted_Cases","max"),
    Week_of_Peak=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(),"Semaine"])
).reset_index()

# ============================================================
# 9. Carte interactive – popups très informatifs
# ============================================================
m = folium.Map(location=[15,8], zoom_start=6)
sa_gdf = sa_gdf.merge(risk_df, left_on="ADM3_NAME", right_on="Aire_Sante", how="left")

import branca.colormap as cm
max_cases = sa_gdf["Max_Predicted_Cases"].max()
colormap = cm.linear.OrRd_09.scale(0, max_cases)
colormap.caption = "Cas rouges prévus sur 12 semaines"
colormap.add_to(m)

def popup_html(row):
    return f"""
    <b>{row['ADM3_NAME']}</b><br>
    Cas observés: {int(row.get('Cas_Observes',0))}<br>
    Cas prévus max: {int(row.get('Max_Predicted_Cases',0))}<br>
    Semaine de pic: {row.get('Week_of_Peak','-')}<br>
    Population enfants 0-4 ans: {int(row.get('Pop_Totale',0))}<br>
    Urbanisation: {row.get('Urbanisation','-')}<br>
    Coef climatique: {row.get('Coef_Climatique',0):.2f}
    """

folium.GeoJson(
    sa_gdf,
    style_function=lambda feature:{
        "fillColor":colormap(feature["properties"]["Max_Predicted_Cases"]),
        "color":"black","weight":1,"fillOpacity":0.7
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["ADM3_NAME","Max_Predicted_Cases","Week_of_Peak"],
        aliases=["Aire de santé","Cas max prévus","Semaine de pic"],
        localize=True
    ),
    popup=folium.GeoJsonPopup(fields=[], labels=False, parse_html=True,
                              html=[popup_html(r) for _,r in sa_gdf.iterrows()])
).add_to(m)

st.subheader("🗺️ Carte des aires de santé – risque de rougeole")
st.components.v1.html(m._repr_html_(), height=650)

# ============================================================
# 10. KPIs & tableau de bord
# ============================================================
st.subheader("📊 Tableau de bord – Indicateurs par Aire de Santé")
weekly_kpi = df.groupby("Aire_Sante").agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines_pct=("Statut_Vaccinal", lambda x:(x=="Non").mean()*100),
    Age_Moyen=("Age_Mois","mean")
).reset_index()
st.dataframe(weekly_kpi)

st.subheader("🔮 Cas attendus sur 12 prochaines semaines")
st.line_chart(future_df.pivot(index="Semaine",columns="Aire_Sante",values="Predicted_Cases"))

st.subheader("🚨 Aires de santé – risque maximal sur 12 semaines")
st.dataframe(risk_df.sort_values("Max_Predicted_Cases", ascending=False))
