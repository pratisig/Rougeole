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
import os
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIGURATION & STYLE
# ============================================================
st.set_page_config(page_title="Rougeole Prédiction PRO", layout="wide", page_icon="🦠")

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
# 2. CHARGEMENT SÉCURISÉ DU GEOJSON
# ============================================================
@st.cache_data
def load_geo_data(file_source):
    """Charge le GeoJSON et extrait les stats GEE"""
    try:
        # Lecture du fichier
        if isinstance(file_source, str):
            gdf = gpd.read_file(file_source)
        else:
            gdf = gpd.read_file(file_source)
            
        # Conversion GEE pour extraction socio-démo
        fc = geemap.gdf_to_ee(gdf)
        pop = ee.ImageCollection("WorldPop/GPW/v11/population").first()
        smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")

        def reducer(feature):
            geom = feature.geometry()
            p_val = pop.reduceRegion(ee.Reducer.sum(), geom, 100).get('population')
            u_val = smod.reduceRegion(ee.Reducer.mode(), geom, 1000).get('smod_code')
            return feature.set({'pop_count': p_val, 'urban_code': u_val})

        stats = fc.map(reducer)
        df_stats = geemap.ee_to_pandas(stats)
        
        # Standardisation des colonnes (détection automatique de la colonne nom)
        potential_names = ['health_area', 'Aire_Sante', 'name_fr', 'ADM3_FR']
        found_col = next((c for c in gdf.columns if c in potential_names), gdf.columns[0])
        
        gdf = gdf.rename(columns={found_col: 'Aire_Sante'})
        df_stats = df_stats.rename(columns={found_col: 'Aire_Sante'})
        
        return gdf, df_stats
    except Exception as e:
        st.error(f"Erreur lors de l'analyse géographique : {e}")
        return None, None

# ============================================================
# 3. INTERFACE PRINCIPALE
# ============================================================
st.title("🦠 Surveillance & Prédiction Rougeole (Niger)")

if not init_gee():
    st.stop()

# --- GESTION DU FICHIER GÉOGRAPHIQUE ---
# On vérifie plusieurs noms de fichiers possibles pour éviter l'erreur d'accent
possible_files = ["aire_de_santé.geojson", "aire_de_sante.geojson"]
geo_file_path = next((f for f in possible_files if os.path.exists(f)), None)

if geo_file_path:
    st.sidebar.success(f"✅ Fond de carte détecté : {geo_file_path}")
    base_gdf, data_geo_df = load_geo_data(geo_file_path)
else:
    st.sidebar.warning("⚠️ Fichier 'aire_de_sante.geojson' non trouvé sur le serveur.")
    uploaded_geo = st.sidebar.file_uploader("Veuillez uploader le fichier GeoJSON manuellement", type="geojson")
    if uploaded_geo:
        base_gdf, data_geo_df = load_geo_data(uploaded_geo)
    else:
        st.info("En attente du fichier géographique pour démarrer...")
        st.stop()

# --- CHARGEMENT DU LINELIST ---
uploaded_linelist = st.sidebar.file_uploader("Charger le Linelist (CSV)", type="csv")

if uploaded_linelist and base_gdf is not None:
    df_cases = pd.read_csv(uploaded_linelist, parse_dates=['Date_Debut_Eruption'])
    
    # Prétraitement
    df_cases['Semaine'] = df_cases['Date_Debut_Eruption'].dt.isocalendar().week
    df_cases['Annee'] = df_cases['Date_Debut_Eruption'].dt.year
    
    # Agrégation Hebdomadaire
    weekly_data = df_cases.groupby(['Annee', 'Semaine', 'Aire_Sante']).size().reset_index(name='Cas_Observes')

    # ============================================================
    # 4. MODÉLISATION & PRÉDICTION
    # ============================================================
    with st.spinner("Entraînement du modèle prédictif..."):
        # Fusion données épidémo + socio-démo
        full_df = weekly_data.merge(data_geo_df[['Aire_Sante', 'pop_count', 'urban_code']], on='Aire_Sante', how='left')
        
        le = LabelEncoder()
        full_df['Aire_Encoded'] = le.fit_transform(full_df['Aire_Sante'])
        
        # Features
        features = ['Semaine', 'pop_count', 'urban_code', 'Aire_Encoded']
        X = full_df[features].fillna(0)
        y = full_df['Cas_Observes']

        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(X, y)

        # Projection sur 8 semaines
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

    # ============================================================
    # 5. AFFICHAGE DU DASHBOARD
    # ============================================================
    tab1, tab2 = st.tabs(["🗺️ Cartographie du Risque", "📊 Statistiques"])

    with tab1:
        st.subheader("Prévision des cas par Aire de Santé (8 prochaines semaines)")
        
        # Préparation de la carte
        risk_map = df_pred.groupby('Aire_Sante')['Cas_Prevus'].sum().reset_index()
        map_final = base_gdf.merge(risk_map, on='Aire_Sante', how='left').fillna(0)
        
        # Centrage
        m = folium.Map(location=[13.5, 2.1], zoom_start=7, tiles="cartodbpositron")
        
        folium.Choropleth(
            geo_data=map_final,
            data=map_final,
            columns=['Aire_Sante', 'Cas_Prevus'],
            key_on='feature.properties.Aire_Sante',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            legend_name="Cas cumulés prévus"
        ).add_to(m)
        
        st_folium(m, width=900, height=500)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 🚨 Zones à haut risque")
            st.dataframe(risk_map.sort_values('Cas_Prevus', ascending=False).head(10))
        
        with col2:
            st.write("### 📈 Tendance Globale")
            fig = px.line(df_pred.groupby('Semaine')['Cas_Prevus'].sum().reset_index(), 
                          x='Semaine', y='Cas_Prevus', title="Évolution prévue des cas")
            st.plotly_chart(fig)

else:
    st.info("Veuillez charger votre fichier Linelist (CSV) pour générer les prédictions.")
