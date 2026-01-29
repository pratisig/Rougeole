# ============================================================
# APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
import json
import ee
from streamlit_folium import st_folium
import folium
import branca.colormap as cm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 0. CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Surveillance Rougeole Multi-pays",
    layout="wide",
    page_icon="🦠",
)
st.title("🦠 Dashboard de Surveillance & Prédiction – Rougeole")

# ============================================================
# 1. INITIALISATION GOOGLE EARTH ENGINE
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
    upload_file = st.sidebar.file_uploader("Charger un shapefile/GeoJSON", type=["shp","geojson"])

# Linelist et vaccin
linelist_file = st.sidebar.file_uploader("Linelists rougeole (CSV)", type=["csv"])
vacc_file = st.sidebar.file_uploader("Couverture vaccinale (CSV – optionnel)", type=["csv"])

# Période
start_date = st.sidebar.date_input("Date de début", value=datetime(2024,1,1))
end_date = st.sidebar.date_input("Date de fin", value=datetime.today())

# ============================================================
# 3. FONCTIONS UTILES
# ============================================================

def ee_fc_to_gdf(fc):
    """Convertir FeatureCollection GEE en GeoDataFrame"""
    features = fc.getInfo()["features"]
    gdf = gpd.GeoDataFrame.from_features(features)
    return gdf

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

@st.cache_data
def worldpop_children_stats(gdf):
    # Sélectionner M0-M4 et F0-F4
    pop = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic()
    bands = [f"M{i}" for i in range(5)] + [f"F{i}" for i in range(5)]
    pop_children = pop.select(bands)
    stats = pop_children.reduceRegions(collection=ee.FeatureCollection(gdf.__geo_interface__),
                                       reducer=ee.Reducer.sum(),
                                       scale=100)
    # Convertir en DataFrame
    df = pd.DataFrame([f.get("properties") for f in stats.getInfo()["features"]])
    df = df.rename(columns={"sum": "Pop_0_4ans"})
    return df

@st.cache_data
def urban_classification(gdf):
    smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")
    def classify(feature):
        stats = smod.reduceRegion(ee.Reducer.mode(), feature.geometry(), scale=1000, maxPixels=1e9)
        return feature.set({"SMOD": stats.get("smod")})
    fc = ee.FeatureCollection(gdf.__geo_interface__).map(classify)
    df = pd.DataFrame([f.get("properties") for f in fc.getInfo()["features"]])
    df["Urbanisation"] = df["SMOD"]
    return df[["Urbanisation"]]

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
# 4. CHARGEMENT AIRES DE SANTÉ
# ============================================================

if option_aire == "GAUL Admin3 (GEE)":
    fc = ee.FeatureCollection("FAO/GAUL/2015/level3").filter(ee.Filter.eq("ADM0_NAME",pays_selectionne))
    sa_gdf = ee_fc_to_gdf(fc)
    sa_gdf["ADM3_NAME"] = sa_gdf["ADM3_NAME"].astype(str)
elif option_aire == "Upload Shapefile/GeoJSON":
    if upload_file:
        sa_gdf = gpd.read_file(upload_file)
        # Détection automatique de la colonne aire de santé
        for col in ["ADM3_NAME","health_area","name_fr","Name"]:
            if col in sa_gdf.columns:
                sa_gdf = sa_gdf.rename(columns={col:"ADM3_NAME"})
                break
        else:
            st.error(f"Aucune colonne d'aires de santé trouvée. Colonnes dispo: {sa_gdf.columns.tolist()}")
            st.stop()
        sa_gdf["ADM3_NAME"] = sa_gdf["ADM3_NAME"].astype(str)
    else:
        st.warning("Uploader un shapefile/GeoJSON pour continuer")
        st.stop()

st.success(f"{len(sa_gdf)} aires de santé chargées")

# ============================================================
# 5. CHARGEMENT LINELIST
# ============================================================

if linelist_file:
    df = pd.read_csv(linelist_file, parse_dates=["Date_Debut_Eruption","Date_Notification"])
else:
    st.info("Aucun linelist fourni – données simulées utilisées")
    df = generate_dummy_linelists()

# Filtrage temporel
df = df[(df["Date_Debut_Eruption"] >= pd.to_datetime(start_date)) &
        (df["Date_Debut_Eruption"] <= pd.to_datetime(end_date))]

# Couverture vaccinale
if vacc_file:
    vacc = pd.read_csv(vacc_file)
else:
    vacc = pd.DataFrame({"Aire_Sante":sa_gdf["ADM3_NAME"], "Couverture":np.nan})

# ============================================================
# 6. POPULATION & URBANISATION & CLIMAT
# ============================================================

pop_df = worldpop_children_stats(sa_gdf)
urban_df = urban_classification(sa_gdf)
# Pour l'exemple, climat fictif : à remplacer par fetch_climate_nasa_power si besoin
climate_df = pd.DataFrame({"Aire_Sante":sa_gdf["ADM3_NAME"], "Coef_Climatique":1.0})

# ============================================================
# 7. KPID – Agrégation hebdomadaire
# ============================================================

df["Semaine"] = df["Date_Debut_Eruption"].dt.to_period("W").astype(str)
weekly_features = df.groupby(["Aire_Sante","Semaine"]).agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x=="Non").mean()*100)
).reset_index()

weekly_features = weekly_features.merge(pop_df, left_on="Aire_Sante", right_on="ADM3_NAME", how="left")
weekly_features = weekly_features.merge(urban_df, left_index=True, right_index=True, how="left")
weekly_features = weekly_features.merge(climate_df, on="Aire_Sante", how="left")
weekly_features = weekly_features.merge(vacc, on="Aire_Sante", how="left")

# ============================================================
# 8. PRÉVISION – Gradient Boosting
# ============================================================

le_urban = LabelEncoder()
weekly_features["Urban_Encoded"] = le_urban.fit_transform(weekly_features["Urbanisation"].astype(str))
feature_cols = ["Cas_Observes","Non_Vaccines","Pop_0_4ans","Urban_Encoded","Coef_Climatique"]

X = weekly_features[feature_cols]
y = weekly_features["Cas_Observes"]

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X, y)

# Générer prédictions futures 12 semaines
future_weeks = []
n_weeks = 12
latest_week_idx = len(weekly_features["Semaine"].unique())
for aire in weekly_features["Aire_Sante"].unique():
    aire_row = weekly_features[weekly_features["Aire_Sante"]==aire].iloc[-1]
    for i in range(1,n_weeks+1):
        future_weeks.append({
            "Aire_Sante": aire,
            "Semaine": f"Week_{latest_week_idx+i}",
            "Cas_Observes": aire_row["Cas_Observes"],
            "Non_Vaccines": aire_row["Non_Vaccines"],
            "Pop_0_4ans": aire_row["Pop_0_4ans"],
            "Urban_Encoded": aire_row["Urban_Encoded"],
            "Coef_Climatique": aire_row["Coef_Climatique"]
        })
future_df = pd.DataFrame(future_weeks)
future_df["Predicted_Cases"] = model.predict(future_df[feature_cols])

# Classement par risque
risk_df = future_df.groupby("Aire_Sante").agg(
    Max_Predicted_Cases=("Predicted_Cases","max"),
    Week_of_Peak=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(),"Semaine"])
).reset_index()

# ============================================================
# 9. CARTE INTERACTIVE
# ============================================================

m = folium.Map(location=[15,8], zoom_start=6)
sa_gdf_risk = sa_gdf.merge(risk_df, left_on="ADM3_NAME", right_on="Aire_Sante", how="left")

max_cases = sa_gdf_risk["Max_Predicted_Cases"].max()
colormap = cm.linear.OrRd_09.scale(0,max_cases)
colormap.caption = "Cas rouges prévus sur 12 semaines"
colormap.add_to(m)

def style_function(feature):
    val = feature["properties"].get("Max_Predicted_Cases",0)
    return {
        'fillColor': colormap(val),
        'color':'black',
        'weight':1,
        'fillOpacity':0.7
    }

def popup_html(feature):
    props = feature["properties"]
    html = f"""
    <b>{props.get('ADM3_NAME','-')}</b><br>
    Cas Observés: {props.get('Cas_Observes','-')}<br>
    Non Vaccinés (%): {props.get('Non_Vaccines','-'):.1f}<br>
    Pop 0-4 ans: {props.get('Pop_0_4ans','-')}<br>
    Urbain (SMOD): {props.get('Urbanisation','-')}<br>
    Couverture vaccin: {props.get('Couverture','-')}<br>
    Cas Max prévus: {props.get('Max_Predicted_Cases','-'):.1f}<br>
    Semaine pic: {props.get('Week_of_Peak','-')}
    """
    return folium.Popup(html,max_width=300)

folium.GeoJson(
    sa_gdf_risk,
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(fields=["ADM3_NAME","Max_Predicted_Cases","Week_of_Peak"]),
    popup=popup_html
).add_to(m)

st.subheader("🗺️ Carte des aires de santé – risque de rougeole")
st_folium(m, width=900, height=650)

# ============================================================
# 10. KPI & Tableau de bord
# ============================================================

st.subheader("📊 Indicateurs par Aire de Santé")
kpi_df = weekly_features.groupby("Aire_Sante").agg(
    Cas_Observes=("Cas_Observes","sum"),
    Non_Vaccines_pct=("Non_Vaccines","mean"),
    Pop_0_4ans=("Pop_0_4ans","sum")
).reset_index()
st.dataframe(kpi_df.sort_values("Cas_Observes",ascending=False))

st.subheader("📈 Courbes épidémiques – Observé vs Prévu")
plot_df = pd.concat([
    weekly_features.rename(columns={"Cas_Observes":"Cas_Observes"})[["Semaine","Cas_Observes","Aire_Sante"]],
    future_df.rename(columns={"Predicted_Cases":"Cas_Prevus"})[["Semaine","Cas_Prevus","Aire_Sante"]]
], axis=0)
import plotly.express as px
fig_obs = px.line(plot_df, x="Semaine", y="Cas_Observes", color="Aire_Sante", labels={"Cas_Observes":"Cas Observés"})
fig_pred = px.line(plot_df, x="Semaine", y="Cas_Prevus", color="Aire_Sante", labels={"Cas_Prevus":"Cas Prévus"})
st.plotly_chart(fig_obs, use_container_width=True)
st.plotly_chart(fig_pred, use_container_width=True)

# ============================================================
# 11. Alertes
# ============================================================

alert_df = kpi_df.copy()
alert_df["Alerte_Rouge"] = alert_df["Cas_Observes"] >= np.percentile(alert_df["Cas_Observes"],75)
st.subheader("🚨 Aires de Santé en Alerte Rouge")
st.dataframe(alert_df[alert_df["Alerte_Rouge"]==True])
