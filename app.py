import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import ee
import json
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(page_title="Surveillance Rougeole Multi-pays", layout="wide", page_icon="🦠")
st.title("🦠 Dashboard de Surveillance Prédictive – Rougeole")

# =======================
# 1. GEE INIT
# =======================
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

if not init_gee():
    st.stop()

# =======================
# 2. Sidebar – Sélections
# =======================
pays_selectionne = st.sidebar.selectbox("Pays", ["Niger","Burkina Faso","Mali"])
option_aire = st.sidebar.radio("Source Aires de Santé", ["GAUL Admin3 (GEE)","Upload Shapefile/GeoJSON"])
upload_file = None
if option_aire=="Upload Shapefile/GeoJSON":
    upload_file = st.sidebar.file_uploader("Shapefile/GeoJSON", type=["shp","geojson"])

linelist_file = st.sidebar.file_uploader("Linelists CSV", type=["csv"])
vacc_file = st.sidebar.file_uploader("Vaccination CSV (optionnel)", type=["csv"])
start_date = st.sidebar.date_input("Date début", datetime(2024,1,1))
end_date = st.sidebar.date_input("Date fin", datetime.today())

# =======================
# 3. Aires de santé
# =======================
@st.cache_data
def load_gaul_admin3(pays):
    return ee.FeatureCollection("FAO/GAUL/2015/level3").filter(ee.Filter.eq("ADM0_NAME", pays))

if option_aire=="GAUL Admin3 (GEE)":
    gaul_fc = load_gaul_admin3(pays_selectionne)
elif option_aire=="Upload Shapefile/GeoJSON":
    if upload_file:
        sa_gdf = gpd.read_file(upload_file)
    else:
        st.warning("Uploader un fichier pour continuer")
        st.stop()

# =======================
# 4. Linelist
# =======================
@st.cache_data
def generate_dummy_linelists(n=400, aires=[]):
    np.random.seed(42)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0,180,n), unit="D")
    return pd.DataFrame({
        "ID_Cas": range(1,n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(np.random.randint(1,5,n), unit="D"),
        "Aire_Sante": np.random.choice(aires,n) if aires else np.random.choice(["Niamey","Maradi"],n),
        "Age_Mois": np.random.randint(6,180,n),
        "Statut_Vaccinal": np.random.choice(["Oui","Non"], n, p=[0.6,0.4])
    })

if linelist_file:
    df = pd.read_csv(linelist_file, parse_dates=["Date_Debut_Eruption","Date_Notification"])
else:
    aires_list = sa_gdf['ADM3_NAME'].tolist() if option_aire=="Upload Shapefile/GeoJSON" else ["Niamey","Maradi","Zinder"]
    df = generate_dummy_linelists(aires=aires_list)

df = df[(df["Date_Debut_Eruption"]>=pd.to_datetime(start_date)) & (df["Date_Debut_Eruption"]<=pd.to_datetime(end_date))]
df["Semaine"] = df["Date_Debut_Eruption"].dt.to_period("W").astype(str)

# =======================
# 5. Préparation features
# =======================
weekly_features = df.groupby(["Aire_Sante","Semaine"]).agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x=="Non").mean()*100)
).reset_index()

# Population fictive
weekly_features["Pop_0_4"] = np.random.randint(500,5000,len(weekly_features))
weekly_features["Urban_Encoded"] = np.random.randint(0,2,len(weekly_features))

feature_cols = ["Cas_Observes","Non_Vaccines","Pop_0_4","Urban_Encoded"]
X = weekly_features[feature_cols]
y = weekly_features["Cas_Observes"]

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X,y)

# Prévision 12 semaines
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
            "Pop_0_4":aire_row["Pop_0_4"],
            "Urban_Encoded":aire_row["Urban_Encoded"]
        })

future_df = pd.DataFrame(future_weeks)
future_df["Predicted_Cases"] = model.predict(future_df[feature_cols])

risk_df = future_df.groupby("Aire_Sante").agg(
    Max_Predicted_Cases=("Predicted_Cases","max"),
    Week_of_Peak=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(),"Semaine"])
).reset_index()

# =======================
# 6. Carte Folium simplifiée
# =======================
st.subheader("🗺️ Carte – Risque rougeole")
m = folium.Map(location=[15,8], zoom_start=6)
if option_aire=="Upload Shapefile/GeoJSON":
    sa_merge = sa_gdf.merge(risk_df,left_on="ADM3_NAME",right_on="Aire_Sante",how="left")
    colormap = cm.linear.OrRd_09.scale(0, sa_merge["Max_Predicted_Cases"].max())
    colormap.caption = "Cas rouges sur 12 semaines"
    colormap.add_to(m)
    folium.GeoJson(
        sa_merge,
        style_function=lambda f:{'fillColor': colormap(f['properties']['Max_Predicted_Cases']),
                                'color':'black','weight':1,'fillOpacity':0.7},
        tooltip=folium.GeoJsonTooltip(fields=["ADM3_NAME","Max_Predicted_Cases","Week_of_Peak"])
    ).add_to(m)
st_folium(m,width=900,height=650)

# =======================
# 7. Courbes
# =======================
st.subheader("📈 Courbes Observé vs Prévu")
plot_df = pd.concat([
    weekly_features[["Semaine","Cas_Observes","Aire_Sante"]],
    future_df.rename(columns={"Predicted_Cases":"Cas_Prevus"})[["Semaine","Cas_Prevus","Aire_Sante"]]
], axis=0)
fig_obs = px.line(plot_df, x="Semaine", y="Cas_Observes", color="Aire_Sante", labels={"Cas_Observes":"Cas Observés"})
fig_pred = px.line(plot_df, x="Semaine", y="Cas_Prevus", color="Aire_Sante", labels={"Cas_Prevus":"Cas Prévus"})
st.plotly_chart(fig_obs,use_container_width=True)
st.plotly_chart(fig_pred,use_container_width=True)

# =======================
# 8. Tableau KPI
# =======================
st.subheader("🚨 Aires de santé – risque maximal")
st.dataframe(risk_df.sort_values("Max_Predicted_Cases",ascending=False))
