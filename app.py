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
import plotly.express as px

# ============================================================
# 0. CONFIGURATION
# ============================================================
st.set_page_config(page_title="Measles Predict PRO", layout="wide", page_icon="🦠")

@st.cache_resource
def init_gee():
    try:
        # Priorité aux secrets Streamlit
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
        else:
            ee.Initialize() # Local
        return True
    except Exception as e:
        st.error(f"Erreur GEE : {e}")
        return False

if not init_gee(): st.stop()

# ============================================================
# 1. FONCTIONS GEE OPTIMISÉES (SANS BOUCLES LENTES)
# ============================================================

@st.cache_data
def get_socio_demo_stats(gdf_json):
    """
    Calcule Pop et Urbanisme en UNE SEULE FOIS pour éviter le lag.
    """
    # Conversion du GeoJSON string en FeatureCollection
    gdf = gpd.GeoDataFrame.from_features(json.loads(gdf_json))
    fc = ee.FeatureCollection(json.loads(gdf_json))

    # A. POPULATION 0-4 ANS (WorldPop)
    pop = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic()
    bands = [f"M_{i}" for i in range(5)] + [f"F_{i}" for i in range(5)]
    # On somme les bandes AVANT de réduire pour avoir un seul chiffre par polygone
    pop_total_04 = pop.select(bands).reduce(ee.Reducer.sum())

    # B. URBANISATION (GHSL)
    urban = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")

    # Réduction par régions (Beaucoup plus rapide que .getInfo dans une boucle)
    combined_img = pop_total_04.addBands(urban)
    stats = combined_img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(), # Mean pour pop, Mode serait mieux pour SMOD mais mean simplifie ici
        scale=100
    )

    # Export vers Pandas proprement
    features = stats.getInfo()['features']
    data = [f['properties'] for f in features]
    df_gee = pd.DataFrame(data)
    
    # Nettoyage des noms de colonnes GEE (souvent 'sum' ou 'mean')
    df_gee = df_gee.rename(columns={'sum': 'Pop_04_ans', 'smod_code': 'Urbanisation'})
    return df_gee

# ============================================================
# 2. INTERFACE & CHARGEMENT
# ============================================================
st.sidebar.header("📂 Paramètres")
pays = st.sidebar.selectbox("Pays", ["Niger", "Burkina Faso", "Mali"])
option_geo = st.sidebar.radio("Source Géo", ["GAUL Admin3", "Upload GeoJSON"])

# Chargement Géo
if option_geo == "GAUL Admin3":
    # Correction : GAUL peut être instable, on ajoute un try-except
    try:
        fc_gaul = ee.FeatureCollection("FAO/GAUL/2015/level3").filter(ee.Filter.eq("ADM0_NAME", pays))
        sa_gdf = gpd.GeoDataFrame.from_features(fc_gaul.getInfo())
        sa_gdf = sa_gdf.rename(columns={'ADM3_NAME': 'Aire_Sante'})
    except:
        st.error("L'asset GAUL est indisponible. Veuillez uploader un GeoJSON.")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Fichier GeoJSON", type="geojson")
    if uploaded_file:
        sa_gdf = gpd.read_file(uploaded_file)
        # Détection colonne nom
        col = next((c for c in sa_gdf.columns if c.lower() in ['aire_sante', 'health_area', 'adm3_name']), sa_gdf.columns[0])
        sa_gdf = sa_gdf.rename(columns={col: 'Aire_Sante'})
    else:
        st.stop()

# ============================================================
# 3. TRAITEMENT DES DONNÉES & ML
# ============================================================
with st.spinner("Analyse spatiale en cours (GEE)..."):
    # On passe le GeoJSON en string pour le cache Streamlit
    gee_stats = get_socio_demo_stats(sa_gdf.to_json())
    sa_gdf = sa_gdf.merge(gee_stats[['Aire_Sante', 'Pop_04_ans', 'smod_code']], on='Aire_Sante', how='left').fillna(0)

# Linelist (Demo ou Upload)
linelist_file = st.sidebar.file_uploader("Linelist (CSV)", type="csv")
if linelist_file:
    df_epi = pd.read_csv(linelist_file, parse_dates=['Date_Debut_Eruption'])
else:
    # Génération cohérente
    dates = pd.date_range(end=datetime.now(), periods=100)
    df_epi = pd.DataFrame({
        'Date_Debut_Eruption': np.random.choice(dates, 500),
        'Aire_Sante': np.random.choice(sa_gdf['Aire_Sante'].unique(), 500),
        'Statut_Vaccinal': np.random.choice(['Oui', 'Non'], 500)
    })

# Agrégation Hebdo
df_epi['Semaine'] = df_epi['Date_Debut_Eruption'].dt.isocalendar().week
weekly_agg = df_epi.groupby(['Aire_Sante', 'Semaine']).size().reset_index(name='Cas_Observes')

# Fusion finale pour le ML
ml_df = weekly_agg.merge(sa_gdf[['Aire_Sante', 'Pop_04_ans', 'smod_code']], on='Aire_Sante')

# --- MODÈLE ---
le = LabelEncoder()
ml_df['Aire_Enc'] = le.fit_transform(ml_df['Aire_Sante'])

X = ml_df[['Semaine', 'Pop_04_ans', 'smod_code', 'Aire_Enc']]
y = ml_df['Cas_Observes']

model = GradientBoostingRegressor(n_estimators=100)
model.fit(X, y)

# Prédiction future (8 semaines)
future_list = []
last_w = ml_df['Semaine'].max()
for aire in ml_df['Aire_Sante'].unique():
    base = ml_df[ml_df['Aire_Sante'] == aire].iloc[-1]
    for i in range(1, 9):
        future_list.append({
            'Aire_Sante': aire,
            'Semaine': (last_w + i),
            'Pop_04_ans': base['Pop_04_ans'],
            'smod_code': base['smod_code'],
            'Aire_Enc': base['Aire_Enc']
        })

df_pred = pd.DataFrame(future_list)
df_pred['Cas_Prevus'] = model.predict(df_pred[['Semaine', 'Pop_04_ans', 'smod_code', 'Aire_Enc']]).clip(0)

# ============================================================
# 4. DASHBOARD & CARTE
# ============================================================
st.header(f"📊 Dashboard Rougeole - {pays}")

c1, c2 = st.columns([2, 1])

with c1:
    # Carte Folium
    m = folium.Map(location=[sa_gdf.geometry.centroid.y.mean(), sa_gdf.geometry.centroid.x.mean()], zoom_start=6)
    
    # Données pour la carte (cumul des prédictions)
    map_data = df_pred.groupby('Aire_Sante')['Cas_Prevus'].sum().reset_index()
    sa_gdf_map = sa_gdf.merge(map_data, on='Aire_Sante', how='left').fillna(0)
    
    folium.Choropleth(
        geo_data=sa_gdf_map,
        data=sa_gdf_map,
        columns=['Aire_Sante', 'Cas_Prevus'],
        key_on='feature.properties.Aire_Sante',
        fill_color='YlOrRd',
        legend_name="Cas prévus (8 sem)"
    ).add_to(m)
    
    st_folium(m, width=800, height=500)

with c2:
    st.subheader("🚨 Zones à risque")
    st.dataframe(map_data.sort_values('Cas_Prevus', ascending=False).head(10))

# Graphique Temporel
st.subheader("📈 Tendances par Aire de Santé")
fig = px.line(df_pred, x='Semaine', y='Cas_Prevus', color='Aire_Sante')
st.plotly_chart(fig, use_container_width=True)
