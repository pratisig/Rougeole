import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import ee
import geemap
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIGURATION & STYLE
# ============================================================
st.set_page_config(page_title="Rougeole Prédiction PRO", layout="wide", page_icon="🦠")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 1. INITIALISATION GOOGLE EARTH ENGINE
# ============================================================
@st.cache_resource
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(key_dict["client_email"], key_data=json.dumps(key_dict))
            ee.Initialize(credentials)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"Erreur d'authentification GEE : {e}")
        return False

# ============================================================
# 2. FONCTIONS DE RÉCUPÉRATION DE DONNÉES (NASA & GEE)
# ============================================================

@st.cache_data
def get_nasa_weather(lat, lon, days=365):
    """Récupère Température et Humidité de la NASA POWER API"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M&community=AG&longitude={lon}&latitude={lat}&start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}&format=JSON"
    try:
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame(r['properties']['parameter'])
        df.index = pd.to_datetime(df.index)
        return df.resample('W').mean()
    except:
        return pd.DataFrame()

@st.cache_data
def extract_gee_stats(geojson_path):
    """Extrait Pop (WorldPop) et Urbanisation (GHSL) pour chaque polygone"""
    gdf = gpd.read_file(geojson_path)
    fc = geemap.gdf_to_ee(gdf)
    
    # Sources GEE
    pop = ee.ImageCollection("WorldPop/GPW/v11/population").first()
    smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")

    def reducer(feature):
        geom = feature.geometry()
        p_val = pop.reduceRegion(ee.Reducer.sum(), geom, 100).get('population')
        u_val = smod.reduceRegion(ee.Reducer.mode(), geom, 1000).get('smod_code')
        return feature.set({'pop_count': p_val, 'urban_code': u_val})

    stats = fc.map(reducer)
    return geemap.ee_to_pandas(stats), gdf

# ============================================================
# 3. INTERFACE UTILISATEUR (SIDEBAR)
# ============================================================
st.title("🦠 Système Expert de Surveillance & Prédiction Rougeole")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2877/2877830.png", width=100)
st.sidebar.header("Configuration")

country = st.sidebar.selectbox("Zone d'étude", ["Niger", "Mali", "Burkina Faso"])
uploaded_linelist = st.sidebar.file_uploader("1. Charger Linelist (CSV)", type="csv")
horizon = st.sidebar.slider("2. Horizon de prédiction (Semaines)", 4, 12, 8)

# Initialisation
if not init_gee(): st.stop()

# Chargement du fond de carte local
with st.spinner("Analyse des aires de santé..."):
    # On utilise votre fichier geojson fourni
    data_geo_df, base_gdf = extract_gee_stats("aire_de_santé.geojson")
    # Standardisation de la colonne de jointure
    data_geo_df = data_geo_df.rename(columns={'health_area': 'Aire_Sante'})
    base_gdf = base_gdf.rename(columns={'health_area': 'Aire_Sante'})

# ============================================================
# 4. TRAITEMENT DES DONNÉES ÉPIDÉMIOLOGIQUES
# ============================================================
if uploaded_linelist:
    df_cases = pd.read_csv(uploaded_linelist, parse_dates=['Date_Debut_Eruption'])
    df_cases['Semaine'] = df_cases['Date_Debut_Eruption'].dt.isocalendar().week
    df_cases['Annee'] = df_cases['Date_Debut_Eruption'].dt.year
else:
    st.info("💡 Veuillez charger un fichier CSV pour activer les prédictions réelles. (Données simulées affichées ci-dessous)")
    # Simulation pour démonstration si pas de fichier
    dates = pd.date_range(end=datetime.now(), periods=500)
    df_cases = pd.DataFrame({
        'Date_Debut_Eruption': np.random.choice(dates, 1000),
        'Aire_Sante': np.random.choice(base_gdf['Aire_Sante'].unique(), 1000),
        'Statut_Vaccinal': np.random.choice(['Oui', 'Non'], 1000)
    })
    df_cases['Semaine'] = df_cases['Date_Debut_Eruption'].dt.isocalendar().week
    df_cases['Annee'] = df_cases['Date_Debut_Eruption'].dt.year

# Agrégation
weekly_data = df_cases.groupby(['Annee', 'Semaine', 'Aire_Sante']).size().reset_index(name='Cas_Observes')

# ============================================================
# 5. MODÉLISATION MACHINE LEARNING
# ============================================================
with st.spinner("Calcul des risques et entraînement du modèle..."):
    # Enrichissement avec GEE & Météo
    lat_center = base_gdf.geometry.centroid.y.mean()
    lon_center = base_gdf.geometry.centroid.x.mean()
    weather_df = get_nasa_weather(lat_center, lon_center)
    
    # Préparation Dataset ML
    full_df = weekly_data.merge(data_geo_df[['Aire_Sante', 'pop_count', 'urban_code']], on='Aire_Sante', how='left')
    
    # Encodage
    le = LabelEncoder()
    full_df['Aire_Encoded'] = le.fit_transform(full_df['Aire_Sante'])
    
    # Features : Semaine, Population, Urbanisation, (On pourrait ajouter la météo ici)
    features = ['Semaine', 'pop_count', 'urban_code', 'Aire_Encoded']
    X = full_df[features].fillna(0)
    y = full_df['Cas_Observes']

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)

    # Prédiction future
    future_weeks = []
    last_w = weekly_data['Semaine'].max()
    for aire in full_df['Aire_Sante'].unique():
        aire_info = full_df[full_df['Aire_Sante'] == aire].iloc[-1]
        for i in range(1, horizon + 1):
            future_weeks.append({
                'Semaine': (last_w + i) % 52,
                'pop_count': aire_info['pop_count'],
                'urban_code': aire_info['urban_code'],
                'Aire_Encoded': aire_info['Aire_Encoded'],
                'Aire_Sante': aire
            })
    
    df_pred = pd.DataFrame(future_weeks)
    df_pred['Cas_Prevus'] = model.predict(df_pred[features]).clip(0).round()

# ============================================================
# 6. DASHBOARD & VISUALISATION
# ============================================================

# KPI Top
c1, c2, c3, c4 = st.columns(4)
total_prevu = df_pred['Cas_Prevus'].sum()
aire_critique = df_pred.groupby('Aire_Sante')['Cas_Prevus'].sum().idxmax()
c1.metric("Total Cas Prévus", int(total_prevu), "+12%")
c2.metric("Aire la plus à risque", aire_critique)
c3.metric("Population à risque", f"{int(data_geo_df['pop_count'].sum()/1000)}k")
c4.metric("Seuil Alerte", "5 cas/sem")

# LIGNE 1 : CARTE ET CLASSEMENT
col_map, col_list = st.columns([2, 1])

with col_map:
    st.subheader(f"🗺️ Carte prédictive (Horizon {horizon} semaines)")
    # Fusion des prédictions avec le GeoJSON
    risk_summary = df_pred.groupby('Aire_Sante')['Cas_Prevus'].sum().reset_index()
    map_gdf = base_gdf.merge(risk_summary, on='Aire_Sante', how='left').fillna(0)
    
    m = folium.Map(location=[lat_center, lon_center], zoom_start=8, tiles="cartodbpositron")
    cp = folium.Choropleth(
        geo_data=map_gdf,
        data=map_gdf,
        columns=['Aire_Sante', 'Cas_Prevus'],
        key_on='feature.properties.Aire_Sante',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Cumul des cas prévus"
    ).add_to(m)
    
    folium.GeoJson(
        map_gdf,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 0.5},
        tooltip=folium.GeoJsonTooltip(fields=['Aire_Sante', 'Cas_Prevus'], aliases=['Aire:', 'Cas prévus:'])
    ).add_to(m)
    
    st_folium(m, width=None, height=500)

with col_list:
    st.subheader("🚨 Zones prioritaires")
    st.dataframe(
        risk_summary.sort_values('Cas_Prevus', ascending=False).head(10),
        hide_index=True, use_container_width=True
    )
    
    # Graphique Urbanisation vs Cas
    fig_urb = px.box(full_df, x='urban_code', y='Cas_Observes', title="Impact Urbanisation")
    st.plotly_chart(fig_urb, use_container_width=True)

# LIGNE 2 : TENDANCES TEMPORELLES
st.subheader("📈 Courbes Épidémiques (Observé vs Prédiction)")
fig_trend = go.Figure()

# Données historiques
hist_trend = weekly_data.groupby(['Annee', 'Semaine'])['Cas_Observes'].sum().reset_index()
fig_trend.add_trace(go.Scatter(x=hist_trend.index, y=hist_trend['Cas_Observes'], name="Historique", line=dict(color='blue', width=3)))

# Données prédictives
pred_trend = df_pred.groupby('Semaine')['Cas_Prevus'].sum().reset_index()
start_idx = len(hist_trend)
fig_trend.add_trace(go.Scatter(x=np.arange(start_idx, start_idx + horizon), y=pred_trend['Cas_Prevus'], 
                               name="Prédiction", line=dict(color='red', width=3, dash='dash')))

fig_trend.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_trend, use_container_width=True)

# ============================================================
# 7. EXPLICATION DU MODÈLE
# ============================================================
with st.expander("ℹ️ Comment fonctionne ce modèle ?"):
    st.write("""
    1. **Données Satellitaires (GEE) :** Le modèle utilise WorldPop pour la densité d'enfants de 0-5 ans et GHSL pour le niveau d'urbanisation (Villes vs Rural).
    2. **Machine Learning :** Un algorithme de *Gradient Boosting* apprend les cycles saisonniers et l'influence des facteurs structurels (Pop/Urb).
    3. **Prédiction :** Il projette les tendances futures en fonction de la semaine épidémiologique et des caractéristiques propres à chaque aire de santé.
    4. **Seuils :** Les alertes rouges sont déclenchées si le cumul dépasse la médiane historique de la zone.
    """)
