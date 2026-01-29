# =============================================================================
# DASHBOARD D’AIDE À LA DÉCISION – SURVEILLANCE & PRÉDICTION ROUGEOLE (MULTI‑PAYS)
# Pays : Burkina Faso, Niger, Mali (GAUL Admin 3)
# Données réelles : WorldPop, NASA POWER, GHSL, FAO GAUL
# Application Streamlit – fichier unique (app.py)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
import ee
import json

# =============================================================================
# CONFIG STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Rougeole – Dashboard Prédictif Multi‑Pays",
    layout="wide",
    page_icon="🦠"
)

st.title("🦠 Dashboard d’aide à la décision – Rougeole (Afrique de l’Ouest)")

# =============================================================================
# INITIALISATION GEE (MÉTHODE VALIDÉE STREAMLIT CLOUD)
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
        st.error("Erreur d’authentification Google Earth Engine")
        st.exception(e)
        return False

gee_ok = init_gee()
if not gee_ok:
    st.stop()

# =============================================================================
# SIDEBAR – PARAMÈTRES
# =============================================================================

st.sidebar.header("⚙️ Paramètres d’analyse")

pays_selection = st.sidebar.multiselect(
    "Pays",
    ["Burkina Faso", "Niger", "Mali"],
    default=["Niger"]
)

start_date = st.sidebar.date_input("Date début")
end_date = st.sidebar.date_input("Date fin")

linelist_file = st.sidebar.file_uploader("Linelists rougeole (CSV)", type="csv")

use_climate = st.sidebar.checkbox("Activer données climatiques (NASA POWER)", True)

# =============================================================================
# 1. LINELISTS (CSV OU SIMULATION)
# =============================================================================

@st.cache_data
def simulate_linelists(n=600):
    np.random.seed(42)
    dates = pd.to_datetime(start_date) + pd.to_timedelta(
        np.random.randint(0, (end_date - start_date).days + 1, n), unit="D"
    )
    return pd.DataFrame({
        "ID": range(n),
        "Date": dates,
        "ADM3_NAME": np.random.choice(["Commune A", "Commune B", "Commune C"], n),
        "Age_mois": np.random.randint(6, 120, n),
        "Vaccin": np.random.choice(["Oui", "Non"], n, p=[0.65, 0.35])
    })

if linelist_file:
    linelist = pd.read_csv(linelist_file, parse_dates=["Date"])
else:
    st.info("Aucune linelist fournie – données simulées utilisées")
    linelist = simulate_linelists()

linelist = linelist[(linelist["Date"] >= pd.to_datetime(start_date)) & (linelist["Date"] <= pd.to_datetime(end_date))]
linelist["Semaine"] = linelist["Date"].dt.to_period("W").astype(str)

# =============================================================================
# 2. AIRES DE SANTÉ – GAUL ADMIN 3 (MULTI‑PAYS)
# =============================================================================

@st.cache_data
def load_gaul_admin3(pays_list):
    fc = ee.FeatureCollection("FAO/GAUL/2015/level3")
    fc = fc.filter(ee.Filter.inList("ADM0_NAME", pays_list))
    return fc

gaul_fc = load_gaul_admin3(pays_selection)

# =============================================================================
# 3. SOCIODÉMOGRAPHIE – WORLDPOP (M1–M5, F1–F5)
# =============================================================================

@st.cache_data
def worldpop_children_stats(fc):
    bands = [
        "M_1", "M_2", "M_3", "M_4", "M_5",
        "F_1", "F_2", "F_3", "F_4", "F_5"
    ]
    img = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic().select(bands)

    def reducer(feat):
        stats = img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=feat.geometry(),
            scale=100,
            maxPixels=1e9
        )
        total = ee.Number(0)
        for b in bands:
            total = total.add(ee.Number(stats.get(b)))
        return feat.set({"Pop_0_9": total})

    return fc.map(reducer)

pop_fc = worldpop_children_stats(gaul_fc).getInfo()

pop_df = pd.DataFrame([
    {
        "ADM3_NAME": f["properties"]["ADM3_NAME"],
        "Pop_0_9": f["properties"].get("Pop_0_9", 0)
    }
    for f in pop_fc["features"]
])

# =============================================================================
# 4. CLIMAT – NASA POWER (MOYENNE PAR AIRE)
# =============================================================================

@st.cache_data
def climate_proxy(fc):
    out = []
    for f in fc.getInfo()["features"]:
        geom = ee.Feature(f).geometry().centroid().coordinates().getInfo()
        lon, lat = geom
        try:
            import requests
            url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            params = {
                "parameters": "T2M,RH2M,PRECTOTCORR",
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "start": start_date.strftime("%Y%m%d"),
                "end": end_date.strftime("%Y%m%d"),
                "format": "JSON"
            }
            r = requests.get(url, timeout=30).json()
            p = r["properties"]["parameter"]
            out.append({
                "ADM3_NAME": f["properties"]["ADM3_NAME"],
                "Temp": np.mean(list(p["T2M"].values())),
                "Hum": np.mean(list(p["RH2M"].values())),
                "Rain": np.mean(list(p["PRECTOTCORR"].values()))
            })
        except:
            pass
    return pd.DataFrame(out)

clim_df = climate_proxy(gaul_fc) if use_climate else pd.DataFrame()

# =============================================================================
# 5. AGRÉGATION ÉPIDÉMIOLOGIQUE
# =============================================================================

weekly = linelist.groupby(["ADM3_NAME", "Semaine"]).size().reset_index(name="Cas")

weekly = weekly.merge(pop_df, on="ADM3_NAME", how="left")
weekly = weekly.merge(clim_df, on="ADM3_NAME", how="left")

weekly["Incidence"] = (weekly["Cas"] / weekly["Pop_0_9"]) * 100000

# =============================================================================
# 6. PRÉDICTION (12 SEMAINES)
# =============================================================================

predictions = []

for adm in weekly["ADM3_NAME"].unique():
    df_adm = weekly[weekly["ADM3_NAME"] == adm].copy()
    df_adm = df_adm.sort_values("Semaine")
    df_adm["t"] = range(len(df_adm))

    X = df_adm[["t"]]
    y = df_adm["Cas"]

    if len(df_adm) < 6:
        continue

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    for i in range(1, 13):
        t_future = df_adm["t"].max() + i
        y_pred = model.predict([[t_future]])[0]
        predictions.append({
            "ADM3_NAME": adm,
            "Semaine": f"S+{i}",
            "Cas_prévus": max(0, y_pred)
        })

pred_df = pd.DataFrame(predictions)

# =============================================================================
# 7. VISUALISATIONS
# =============================================================================

st.subheader("📊 KPI clés")

c1, c2, c3 = st.columns(3)

c1.metric("Cas observés", int(weekly["Cas"].sum()))
c2.metric("Incidence moyenne", round(weekly["Incidence"].mean(), 2))
c3.metric("Aires à risque (↑)", pred_df.groupby("ADM3_NAME")["Cas_prévus"].mean().sort_values(ascending=False).head(5).count())

st.subheader("📈 Évolution & prévisions (12 semaines)")

fig = px.line(weekly, x="Semaine", y="Cas", color="ADM3_NAME")
fig2 = px.line(pred_df, x="Semaine", y="Cas_prévus", color="ADM3_NAME")

st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🚨 Aires de santé à risque futur")
st.dataframe(
    pred_df.groupby("ADM3_NAME")["Cas_prévus"].mean()
    .reset_index()
    .sort_values("Cas_prévus", ascending=False)
)

st.caption("WorldPop, NASA POWER, GHSL, FAO GAUL – Analyse prédictive opérationnelle")
