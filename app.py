import streamlit as st
import pandas as pd
import geopandas as gpd
import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import folium
import json
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Surveillance Rougeole - PRO", layout="wide")

@st.cache_resource
def init_gee():
    try:
        # Priorité aux secrets Streamlit pour la prod
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(key_dict["client_email"], key_data=json.dumps(key_dict))
            ee.Initialize(credentials)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"Erreur GEE : {e}")
        return False

if not init_gee():
    st.stop()

# --- CHARGEMENT DES FRONTIÈRES (CORRECTIONS) ---

@st.cache_data
def load_boundaries(source_type, _uploaded_file=None):
    """
    Charge les limites administratives. 
    Source de secours : geoBoundaries (plus fiable que GAUL en ce moment).
    """
    try:
        if source_type == "Fichier Local (Auto)":
            # Tente de charger le fichier que vous avez déjà fourni
            if os.path.exists("aire_de_santé.geojson"):
                gdf = gpd.read_file("aire_de_santé.geojson")
                return gdf
            else:
                st.error("Fichier 'aire_de_santé.geojson' introuvable.")
                return None
        
        elif source_type == "geoBoundaries (ADM3)":
            # Alternative fiable à FAO/GAUL
            url = "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/NER/ADM3/geoBoundaries-NER-ADM3.geojson"
            return gpd.read_file(url)
            
        elif source_type == "Uploader un fichier" and _uploaded_file is not None:
            return gpd.read_file(_uploaded_file)
            
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        return None

# --- INTERFACE SIDEBAR ---
st.sidebar.title("Configuration des Données")
option = st.sidebar.selectbox("Source des Aires de Santé", 
                              ["Fichier Local (Auto)", "geoBoundaries (ADM3)", "Uploader un fichier"])

uploaded_geo = None
if option == "Uploader un fichier":
    uploaded_geo = st.sidebar.file_uploader("Joindre GeoJSON ou Shapefile", type=["geojson", "json", "zip"])

# Chargement du GeoDataFrame
gdf_areas = load_boundaries(option, uploaded_geo)

# --- VALIDATION ET AFFICHAGE ---
if gdf_areas is not None:
    # Standardisation du nom de la colonne
    # On cherche 'health_area' ou 'shapeName' ou 'name'
    cols = gdf_areas.columns.tolist()
    target_col = next((c for c in cols if c in ['health_area', 'shapeName', 'ADM3_FR', 'name_fr']), cols[0])
    gdf_areas = gdf_areas.rename(columns={target_col: "Aire_Sante"})

    st.success(f"Chargé : {len(gdf_areas)} aires de santé détectées.")

    # Conversion pour GEE (utile pour les futurs calculs socio-démo)
    try:
        ee_fc = geemap.gdf_to_ee(gdf_areas)
    except:
        st.warning("Impossible de convertir en objet GEE, mais l'affichage local fonctionnera.")

    # --- TABS ---
    tab_map, tab_data = st.tabs(["🗺️ Cartographie", "📊 Données"])

    with tab_map:
        # Centrage dynamique
        bounds = gdf_areas.total_bounds
        m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2], zoom_start=7)
        
        # Ajout des polygones
        folium.GeoJson(
            gdf_areas,
            name="Aires de Santé",
            tooltip=folium.GeoJsonTooltip(fields=["Aire_Sante"]),
            style_function=lambda x: {'fillColor': '#2ecc71', 'color': 'black', 'weight': 1, 'fillOpacity': 0.3}
        ).add_to(m)
        
        st_folium(m, width=900, height=600)

    with tab_data:
        st.write("Aperçu des données attributaires :")
        st.dataframe(gdf_areas.drop(columns='geometry'))

else:
    st.info("Veuillez sélectionner une source ou uploader un fichier pour continuer.")
