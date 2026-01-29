import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import ee
import geemap # Import principal pour les données
from geemap import foliumap as gmf # Import séparé pour la carte pour éviter le conflit
import folium
from streamlit_folium import st_folium
import plotly.express as px
import requests
import json
import os
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. INITIALISATION & CONFIGURATION
# ============================================================
st.set_page_config(page_title="Rougeole Niger PRO", layout="wide")

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
# 2. CHARGEMENT GÉOGRAPHIQUE & EXTRACTION GEE
# ============================================================
@st.cache_data
def load_and_analyze_geo(file_path):
    """Charge le GeoJSON et extrait la population/urbanisation via GEE"""
    try:
        # Lecture avec Geopandas
        gdf = gpd.read_file(file_path)
        
        # Conversion en FeatureCollection GEE
        fc = geemap.gdf_to_ee(gdf)
        
        # Sources de données GEE
        pop_img = ee.ImageCollection("WorldPop/GPW/v11/population").first()
        smod_img = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")

        # Fonction de réduction pour chaque aire
        def get_stats(f):
            geom = f.geometry()
            p_val = pop_img.reduceRegion(ee.Reducer.sum(), geom, 100).get('population')
            u_val = smod_img.reduceRegion(ee.Reducer.mode(), geom, 1000).get('smod_code')
            return f.set({'pop_count': p_val, 'urban_code': u_val})

        # Application et conversion vers Pandas
        # Utilisation de la méthode CORRECTE de geemap
        with_stats = fc.map(get_stats)
        df_stats = geemap.ee_to_pandas(with_stats)
        
        # Détection de la colonne Nom
        potential_cols = ['health_area', 'Aire_Sante', 'name_fr', 'ADM3_FR']
        found_col = next((c for c in gdf.columns if c in potential_cols), gdf.columns[0])
        
        gdf = gdf.rename(columns={found_col: 'Aire_Sante'})
        df_stats = df_stats.rename(columns={found_col: 'Aire_Sante'})
        
        return gdf, df_stats
    except Exception as e:
        st.error(f"Erreur lors de l'analyse GEE : {e}")
        return None, None

# ============================================================
# 3. INTERFACE PRINCIPALE
# ============================================================
st.title("🦠 Surveillance & Prédiction Rougeole (Niger)")

if not init_gee():
    st.stop()

# --- RECHERCHE DU FICHIER GÉOGRAPHIQUE ---
# On liste tous les fichiers pour trouver celui qui ressemble au GeoJSON (ignore les accents)
all_files = os.listdir('.')
geo_file = next((f for f in all_files if "aire_de_sant" in f and f.endswith(".geojson")), None)

if geo_file:
    st.sidebar.success(f"✅ Fichier trouvé : {geo_file}")
    with st.spinner("Analyse GEE en cours..."):
        base_gdf, data_geo_df = load_and_analyze_geo(geo_file)
else:
    st.sidebar.error("❌ Fichier GeoJSON introuvable à la racine.")
    uploaded_geo = st.sidebar.file_uploader("Veuillez uploader 'aire_de_santé.geojson'", type="geojson")
    if uploaded_geo:
        base_gdf, data_geo_df = load_and_analyze_geo(uploaded_geo)
    else:
        st.stop()

# --- CHARGEMENT DU LINELIST CSV ---
uploaded_csv = st.sidebar.file_uploader("Charger Linelist (CSV)", type="csv")

if uploaded_csv and base_gdf is not None:
    df_cases = pd.read_csv(uploaded_csv, parse_dates=['Date_Debut_Eruption'])
    df_cases['Semaine'] = df_cases['Date_Debut_Eruption'].dt.isocalendar().week
    df_cases['Annee'] = df_cases['Date_Debut_Eruption'].dt.year
    
    # Agrégation
    weekly_data = df_cases.groupby(['Annee', 'Semaine', 'Aire_Sante']).size().reset_index(name='Cas_Observes')

    # --- ML PRÉDICTION ---
    with st.spinner("Entraînement du modèle..."):
        full_df = weekly_data.merge(data_geo_df[['Aire_Sante', 'pop_count', 'urban_code']], on='Aire_Sante', how='left')
        full_df = full_df.fillna(0)
        
        le = LabelEncoder()
        full_df['Aire_Encoded'] = le.fit_transform(full_df['Aire_Sante'])
        
        features = ['Semaine', 'pop_count', 'urban_code', 'Aire_Encoded']
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(full_df[features], full_df['Cas_Observes'])

        # Projection 8 semaines
        future_data = []
        last_w = weekly_data['Semaine'].max()
        for aire in full_df['Aire_Sante'].unique():
            info = full_df[full_df['Aire_Sante'] == aire].iloc[-1]
            for i in range(1, 9):
                future_data.append({
                    'Semaine': (last_w + i) % 52,
                    'pop_count': info['pop_count'],
                    'urban_code': info['urban_code'],
                    'Aire_Encoded': info['Aire_Encoded'],
                    'Aire_Sante': aire
                })
        df_pred = pd.DataFrame(future_data)
        df_pred['Cas_Prevus'] = model.predict(df_pred[features]).clip(0).round()

    # --- AFFICHAGE ---
    tab1, tab2 = st.tabs(["🗺️ Carte du Risque", "📊 Analyses"])

    with tab1:
        st.subheader("Cas prévus par zone (8 prochaines semaines)")
        risk_summary = df_pred.groupby('Aire_Sante')['Cas_Prevus'].sum().reset_index()
        map_final = base_gdf.merge(risk_summary, on='Aire_Sante', how='left').fillna(0)
        
        # Centrage automatique sur le Niger
        m = folium.Map(location=[17.6, 8.1], zoom_start=6, tiles="cartodbpositron")
        folium.Choropleth(
            geo_data=map_final,
            data=map_final,
            columns=['Aire_Sante', 'Cas_Prevus'],
            key_on='feature.properties.Aire_Sante',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            legend_name="Cumul des cas prévus"
        ).add_to(m)
        st_folium(m, width=900, height=500)

    with tab2:
        st.plotly_chart(px.line(df_pred.groupby('Semaine')['Cas_Prevus'].sum().reset_index(), 
                                x='Semaine', y='Cas_Prevus', title="Tendance nationale prévue"))
else:
    st.info("Veuillez charger le fichier CSV pour voir les prédictions.")
