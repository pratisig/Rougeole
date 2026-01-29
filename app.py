import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import ee
import geemap  # <--- Correction : Import indispensable
import folium
from streamlit_folium import st_folium
import plotly.express as px
import json
import io
import os
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. INITIALISATION & CONFIGURATION
# ============================================================
st.set_page_config(page_title="Rougeole Niger Expert", layout="wide", page_icon="🦠")

@st.cache_resource
def init_gee():
    try:
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            project_id = key_dict.get("project_id") or st.secrets.get("GEE_PROJECT_ID")
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"Erreur GEE : {e}")
        return False

# ============================================================
# 2. EXTRACTION DES DONNÉES SOCIO-DÉMO (GEE)
# ============================================================
@st.cache_data
def get_socio_demo_stats(gdf_json):
    """Calcule Pop et Urbanisme en optimisant le transfert de données"""
    try:
        # Recharger le GeoDataFrame et simplifier pour GEE
        gdf = gpd.read_file(io.StringIO(gdf_json))
        gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)
        
        # Conversion vers Google Earth Engine
        fc = geemap.gdf_to_ee(gdf)

        # 1. Population 0-4 ans (WorldPop)
        pop_col = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic()
        bands = [f"M_{i}" for i in range(5)] + [f"F_{i}" for i in range(5)]
        pop_img = pop_col.select(bands).reduce(ee.Reducer.sum()).rename('Pop_04_ans')

        # 2. Urbanisation (GHSL)
        urban_img = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020").select('smod_code')

        # 3. Réduction spatiale
        combined = pop_img.addBands(urban_img)
        stats = combined.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=100
        )

        # On ne récupère que les propriétés (pas les géométries) pour éviter le timeout
        result = stats.select(['Aire_Sante', 'Pop_04_ans', 'smod_code'], retainGeometry=False).getInfo()
        
        df_res = pd.DataFrame([f['properties'] for f in result['features']])
        return df_res

    except Exception as e:
        st.warning(f"⚠️ Extraction GEE limitée ou échouée : {e}")
        # Retourne un DataFrame vide avec les BONNES colonnes pour éviter le KeyError
        return pd.DataFrame(columns=['Aire_Sante', 'Pop_04_ans', 'smod_code'])

# ============================================================
# 3. APPLICATION PRINCIPALE
# ============================================================
st.title("🦠 Système de Prédiction Épidémique")

if not init_gee():
    st.stop()

# --- Chargement Géographique ---
# Recherche automatique du fichier local
all_files = os.listdir('.')
geo_file = next((f for f in all_files if "aire_de_sant" in f and f.endswith(".geojson")), None)

if geo_file:
    sa_gdf = gpd.read_file(geo_file)
    # Standardisation du nom de colonne
    col_name = next((c for c in sa_gdf.columns if c.lower() in ['aire_sante', 'health_area', 'adm3_name']), sa_gdf.columns[0])
    sa_gdf = sa_gdf.rename(columns={col_name: 'Aire_Sante'})
    
    with st.spinner("Analyse des facteurs de risque (GEE)..."):
        # Calcul des stats via GEE
        gee_stats = get_socio_demo_stats(sa_gdf.to_json())
        
        # Fusion des données (Correction de la KeyError ici)
        if not gee_stats.empty:
            sa_gdf = sa_gdf.merge(gee_stats[['Aire_Sante', 'Pop_04_ans', 'smod_code']], on='Aire_Sante', how='left')
        else:
            # Valeurs par défaut si GEE échoue totalement
            sa_gdf['Pop_04_ans'] = 1000
            sa_gdf['smod_code'] = 11
        
        sa_gdf = sa_gdf.fillna(0)
else:
    st.error("Fichier GeoJSON introuvable. Veuillez placer 'aire_de_sante.geojson' à la racine.")
    st.stop()

# --- Chargement Épidémiologique ---
uploaded_csv = st.sidebar.file_uploader("Charger Linelist (CSV)", type="csv")

if uploaded_csv:
    df_cases = pd.read_csv(uploaded_csv, parse_dates=['Date_Debut_Eruption'])
    df_cases['Semaine'] = df_cases['Date_Debut_Eruption'].dt.isocalendar().week
    
    # Agrégation
    weekly_data = df_cases.groupby(['Aire_Sante', 'Semaine']).size().reset_index(name='Cas_Observes')
    
    # --- Modélisation ML ---
    with st.spinner("Entraînement du modèle..."):
        # Fusion avec les données GEE
        full_df = weekly_data.merge(sa_gdf[['Aire_Sante', 'Pop_04_ans', 'smod_code']], on='Aire_Sante', how='left').fillna(0)
        
        le = LabelEncoder()
        full_df['Aire_Enc'] = le.fit_transform(full_df['Aire_Sante'])
        
        features = ['Semaine', 'Pop_04_ans', 'smod_code', 'Aire_Enc']
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(full_df[features], full_df['Cas_Observes'])

        # Prédiction 8 semaines
        future = []
        last_w = full_df['Semaine'].max()
        for aire in full_df['Aire_Sante'].unique():
            row = full_df[full_df['Aire_Sante'] == aire].iloc[-1]
            for i in range(1, 9):
                future.append({
                    'Aire_Sante': aire, 'Semaine': (last_w + i),
                    'Pop_04_ans': row['Pop_04_ans'], 'smod_code': row['smod_code'],
                    'Aire_Enc': row['Aire_Enc']
                })
        df_pred = pd.DataFrame(future)
        df_pred['Cas_Prevus'] = model.predict(df_pred[features]).clip(0).round()

    # --- Affichage Dashboard ---
    col_map, col_stats = st.columns([2, 1])
    
    with col_map:
        st.subheader("🗺️ Carte du Risque (Prévisions)")
        risk_map = df_pred.groupby('Aire_Sante')['Cas_Prevus'].sum().reset_index()
        sa_gdf_risk = sa_gdf.merge(risk_map, on='Aire_Sante', how='left').fillna(0)
        
        m = folium.Map(location=[sa_gdf.geometry.centroid.y.mean(), sa_gdf.geometry.centroid.x.mean()], zoom_start=7)
        folium.Choropleth(
            geo_data=sa_gdf_risk, data=sa_gdf_risk,
            columns=['Aire_Sante', 'Cas_Prevus'],
            key_on='feature.properties.Aire_Sante',
            fill_color='YlOrRd', legend_name="Cas prévus"
        ).add_to(m)
        st_folium(m, width=700, height=500)

    with col_stats:
        st.subheader("🚨 Zones prioritaires")
        st.dataframe(risk_map.sort_values('Cas_Prevus', ascending=False).head(10), hide_index=True)

    st.subheader("📈 Courbes de tendance")
    st.plotly_chart(px.line(df_pred, x='Semaine', y='Cas_Prevus', color='Aire_Sante'), use_container_width=True)

else:
    st.info("👋 Veuillez charger votre fichier CSV pour activer les prédictions.")
