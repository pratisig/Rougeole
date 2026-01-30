"""
============================================================
APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE (Multi-pays)
Version améliorée avec semaines épidémiologiques et seuils
Partie 1/4 - Configuration et chargement des données
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import ee
import json
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import zipfile
import tempfile
import os
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Surveillance Rougeole Multi-pays",
    layout="wide",
    page_icon="🦠",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        font-weight: bold;
        padding: 5px;
        border-radius: 3px;
    }
    .medium-risk {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 5px;
        border-radius: 3px;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 5px;
        border-radius: 3px;
    }
    .stButton>button {
        width: 100%;
    }
    h1 {
        color: #d32f2f;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 10px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🦠 Dashboard de Surveillance et Prédiction - Rougeole")
st.markdown("### Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques")

# ============================================================
# MAPPING PAYS -> ISO3
# ============================================================
PAYS_ISO3_MAP = {
    "Niger": "NER",
    "Burkina Faso": "BFA",
    "Mali": "MLI",
    "Sénégal": "SEN",
    "Tchad": "TCD",
    "Bénin": "BEN",
    "Togo": "TGO",
    "Côte d'Ivoire": "CIV",
    "Ghana": "GHA",
    "Nigeria": "NGA",
    "Guinée": "GIN",
    "Mauritanie": "MRT"
}

# ============================================================
# 1. INITIALISATION GEE (OPTIONNEL)
# ============================================================
@st.cache_resource
def init_gee():
    """Initialisation Google Earth Engine avec gestion d'erreurs"""
    try:
        key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key_dict["client_email"],
            key_data=json.dumps(key_dict)
        )
        ee.Initialize(credentials)
        return True
    except:
        try:
            ee.Initialize()
            return True
        except:
            return False

gee_ok = init_gee()
if gee_ok:
    st.sidebar.success("✓ GEE connecté")

# ============================================================
# 2. SIDEBAR – CONFIGURATION
# ============================================================
st.sidebar.header("📂 Configuration de l'Analyse")

# Initialiser session_state pour la sélection dynamique
if 'pays_precedent' not in st.session_state:
    st.session_state.pays_precedent = None
if 'sa_gdf_cache' not in st.session_state:
    st.session_state.sa_gdf_cache = None

# Sélection pays
pays_selectionne = st.sidebar.selectbox(
    "🌍 Sélectionner le pays", 
    list(PAYS_ISO3_MAP.keys()),
    key='pays_select'
)

iso3_pays = PAYS_ISO3_MAP[pays_selectionne]

# Détecter changement de pays
pays_change = (st.session_state.pays_precedent != pays_selectionne)
if pays_change:
    st.session_state.pays_precedent = pays_selectionne
    st.session_state.sa_gdf_cache = None
    st.rerun()

# Aires de santé
st.sidebar.subheader("🗺️ Aires de Santé")
option_aire = st.sidebar.radio(
    "Source des données géographiques", 
    ["Fichier local (ao_hlthArea.zip)", "Upload personnalisé"],
    key='option_aire'
)

upload_file = None
if option_aire == "Upload personnalisé":
    upload_file = st.sidebar.file_uploader(
        "Charger un fichier géographique", 
        type=["shp", "geojson", "zip"],
        help="Format: Shapefile ou GeoJSON avec colonnes 'iso3' et 'health_area'"
    )

# Linelist
st.sidebar.subheader("📊 Données Épidémiologiques")
option_linelist = st.sidebar.radio(
    "Source des cas de rougeole",
    ["Upload CSV", "Données fictives (test)"],
    key='option_linelist'
)

linelist_file = None
if option_linelist == "Upload CSV":
    linelist_file = st.sidebar.file_uploader(
        "Linelists rougeole (CSV)", 
        type=["csv"],
        help="Format: ID_Cas, Date_Debut_Eruption, Date_Notification, Aire_Sante, Age_Mois, Statut_Vaccinal"
    )

# Période d'analyse
st.sidebar.subheader("📅 Période d'Analyse")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Date début", value=datetime(2024, 1, 1), key='start_date')
with col2:
    end_date = st.date_input("Date fin", value=datetime.today(), key='end_date')

# Période de prédiction
st.sidebar.subheader("🔮 Paramètres de Prédiction")

# Choix de la période en mois
pred_mois = st.sidebar.slider(
    "Période de prédiction (mois)",
    min_value=1,
    max_value=12,
    value=3,
    help="Nombre de mois à prédire après la dernière semaine de données"
)

# Conversion en semaines
n_weeks_pred = pred_mois * 4  # Approximation 1 mois = 4 semaines

st.sidebar.info(f"📆 Prédiction sur **{n_weeks_pred} semaines épidémiologiques** (~{pred_mois} mois)")

# Seuils personnalisables
st.sidebar.subheader("⚙️ Seuils d'Alerte")

with st.sidebar.expander("Configurer les seuils", expanded=False):
    seuil_baisse = st.slider(
        "Seuil de baisse significative (%)",
        min_value=10,
        max_value=90,
        value=75,
        step=5,
        help="Afficher les aires avec baisse ≥ X% par rapport à la moyenne"
    )
    
    seuil_hausse = st.slider(
        "Seuil de hausse significative (%)",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Afficher les aires avec hausse ≥ X% par rapport à la moyenne"
    )
    
    seuil_alerte_epidemique = st.number_input(
        "Seuil d'alerte épidémique (cas/semaine)",
        min_value=1,
        max_value=100,
        value=5,
        help="Nombre de cas par semaine déclenchant une alerte"
    )

# ============================================================
# 3. FONCTIONS DE CHARGEMENT AIRES DE SANTÉ
# ============================================================

@st.cache_data
def load_health_areas_from_zip(zip_path, iso3_filter):
    """
    Charge les aires de santé depuis ao_hlthArea.zip filtré par pays
    
    Args:
        zip_path: Chemin vers le fichier ZIP
        iso3_filter: Code ISO3 du pays (ex: 'NER', 'BFA')
    
    Returns:
        GeoDataFrame filtré
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extraire le ZIP
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
            
            # Trouver le shapefile
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            
            if not shp_files:
                raise ValueError("Aucun fichier .shp trouvé dans le ZIP")
            
            shp_path = os.path.join(tmpdir, shp_files[0])
            
            # Charger le shapefile complet
            gdf_full = gpd.read_file(shp_path)
            
            # Détecter la colonne ISO3 (plusieurs variantes possibles)
            iso3_col = None
            for col in ['iso3', 'ISO3', 'iso_code', 'ISO_CODE', 'country_iso', 'COUNTRY_ISO']:
                if col in gdf_full.columns:
                    iso3_col = col
                    break
            
            if iso3_col is None:
                st.warning(f"⚠️ Colonne ISO3 non trouvée. Colonnes disponibles: {list(gdf_full.columns)}")
                return gpd.GeoDataFrame()
            
            # Filtrer par pays
            gdf = gdf_full[gdf_full[iso3_col] == iso3_filter].copy()
            
            if gdf.empty:
                st.warning(f"⚠️ Aucune aire de santé trouvée pour {iso3_filter}")
                return gpd.GeoDataFrame()
            
            # Standardiser les noms de colonnes
            # Détecter la colonne du nom de l'aire de santé
            name_col = None
            for col in ['health_area', 'HEALTH_AREA', 'name', 'NAME', 'nom', 'NOM', 'aire_sante']:
                if col in gdf.columns:
                    name_col = col
                    break
            
            if name_col:
                gdf['ADM3_NAME'] = gdf[name_col]
            else:
                # Générer des noms si aucune colonne trouvée
                gdf['ADM3_NAME'] = [f"Aire_{i+1}" for i in range(len(gdf))]
            
            # Nettoyer les géométries invalides
            gdf = gdf[gdf.geometry.is_valid]
            
            # Reprojeter si nécessaire
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            
            return gdf
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier ZIP: {e}")
        return gpd.GeoDataFrame()

def load_shapefile_from_upload(upload_file):
    """Charge un shapefile depuis un upload utilisateur"""
    try:
        if upload_file.name.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, 'upload.zip')
                with open(zip_path, 'wb') as f:
                    f.write(upload_file.getvalue())
                
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(tmpdir)
                    shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                    if shp_files:
                        gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                    else:
                        raise ValueError("Aucun fichier .shp trouvé dans le ZIP")
        else:
            gdf = gpd.read_file(upload_file)
        
        # Standardiser les noms de colonnes
        if "ADM3_NAME" not in gdf.columns:
            for col in ["health_area", "HEALTH_AREA", "name", "NAME", "nom", "NOM"]:
                if col in gdf.columns:
                    gdf["ADM3_NAME"] = gdf[col]
                    break
            else:
                gdf["ADM3_NAME"] = [f"Aire_{i}" for i in range(len(gdf))]
        
        # Nettoyer les géométries
        gdf = gdf[gdf.geometry.is_valid]
        
        # Reprojeter si nécessaire
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
        
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier: {e}")
        return gpd.GeoDataFrame()

# ============================================================
# CHARGEMENT DES AIRES DE SANTÉ
# ============================================================

# Utiliser le cache de session_state si disponible et pays identique
if st.session_state.sa_gdf_cache is not None and not pays_change:
    sa_gdf = st.session_state.sa_gdf_cache
    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (cache)")
else:
    with st.spinner(f"🔄 Chargement des aires de santé pour {pays_selectionne}..."):
        if option_aire == "Fichier local (ao_hlthArea.zip)":
            # Chemin vers le fichier ZIP local
            zip_path = os.path.join("data", "ao_hlthArea.zip")
            
            if not os.path.exists(zip_path):
                st.error(f"❌ Fichier non trouvé: {zip_path}")
                st.info("📁 Veuillez placer le fichier 'ao_hlthArea.zip' dans le dossier 'data/'")
                st.stop()
            
            sa_gdf = load_health_areas_from_zip(zip_path, iso3_pays)
            
            if sa_gdf.empty:
                st.error(f"❌ Impossible de charger les aires pour {pays_selectionne} ({iso3_pays})")
                st.stop()
            else:
                st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées ({iso3_pays})")
                st.session_state.sa_gdf_cache = sa_gdf
                
        elif option_aire == "Upload personnalisé":
            if upload_file is None:
                st.warning("⚠️ Veuillez uploader un fichier shapefile ou GeoJSON")
                st.stop()
            else:
                sa_gdf = load_shapefile_from_upload(upload_file)
                if sa_gdf.empty:
                    st.error("❌ Fichier invalide ou vide")
                    st.stop()
                else:
                    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (upload)")
                    st.session_state.sa_gdf_cache = sa_gdf

# Vérification finale
if sa_gdf.empty or sa_gdf is None:
    st.error("❌ Aucune aire de santé chargée. Impossible de continuer.")
    st.stop()

# Afficher un échantillon des aires chargées
with st.expander(f"📋 Aperçu des {len(sa_gdf)} aires de santé chargées", expanded=False):
    st.dataframe(sa_gdf[['ADM3_NAME', 'geometry']].head(10))
    st.caption(f"Projection: {sa_gdf.crs}")
# ============================================================
# 4. GÉNÉRATION/CHARGEMENT LINELIST
# ============================================================

@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500, start=None, end=None):
    """Génère des linelists réalistes pour tests avec semaines épidémiologiques"""
    np.random.seed(42)
    
    if start is None:
        start = datetime(2024, 1, 1)
    if end is None:
        end = datetime.today()
    
    delta_days = (end - start).days
    
    # Distribution temporelle réaliste (exponentielle pour simuler épidémie)
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.random.exponential(scale=delta_days/3, size=n).clip(0, delta_days).astype(int), 
        unit="D"
    )
    
    df = pd.DataFrame({
        "ID_Cas": range(1, n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(
            np.random.poisson(3, n), unit="D"
        ),
        "Aire_Sante": np.random.choice(_sa_gdf["ADM3_NAME"].unique(), n),
        "Age_Mois": np.random.gamma(shape=2, scale=30, size=n).clip(6, 180).astype(int),
        "Statut_Vaccinal": np.random.choice(["Oui", "Non"], n, p=[0.55, 0.45]),
        "Sexe": np.random.choice(["M", "F"], n),
        "Issue": np.random.choice(["Guéri", "Décédé", "Inconnu"], n, p=[0.92, 0.03, 0.05])
    })
    
    return df

# Chargement des données de cas
with st.spinner("📥 Chargement des données de cas..."):
    if option_linelist == "Upload CSV" and linelist_file:
        try:
            df = pd.read_csv(linelist_file)
            
            # Parser les dates avec gestion d'erreurs
            for col in ["Date_Debut_Eruption", "Date_Notification"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Vérifier les colonnes requises
            required_cols = ["Date_Debut_Eruption", "Aire_Sante", "Age_Mois", "Statut_Vaccinal"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Colonnes manquantes: {missing_cols}")
                st.stop()
            
            st.sidebar.success(f"✓ {len(df)} cas chargés")
        except Exception as e:
            st.error(f"❌ Erreur de lecture du fichier CSV: {e}")
            st.stop()
    else:
        df = generate_dummy_linelists(sa_gdf, start=start_date, end=end_date)
        st.sidebar.info("📊 Données fictives utilisées")

# Filtre temporel
df = df[
    (df["Date_Debut_Eruption"] >= pd.to_datetime(start_date)) & 
    (df["Date_Debut_Eruption"] <= pd.to_datetime(end_date))
].copy()

if len(df) == 0:
    st.warning("⚠️ Aucun cas dans la période sélectionnée")
    st.stop()

# ============================================================
# CALCUL DES SEMAINES ÉPIDÉMIOLOGIQUES
# ============================================================

def calculer_semaine_epidemio(date):
    """
    Calcule la semaine épidémiologique (1-52) selon la norme ISO 8601
    La semaine 1 est celle contenant le premier jeudi de l'année
    """
    return date.isocalendar()[1]

# Ajouter les semaines épidémiologiques
df['Semaine_Epi'] = df['Date_Debut_Eruption'].apply(calculer_semaine_epidemio)
df['Annee'] = df['Date_Debut_Eruption'].dt.year
df['Semaine_Annee'] = df['Annee'].astype(str) + '-S' + df['Semaine_Epi'].astype(str).str.zfill(2)

# Identifier la dernière semaine de données
derniere_semaine_epi = df['Semaine_Epi'].max()
derniere_annee = df['Annee'].max()

st.sidebar.info(f"📅 Dernière semaine: **S{derniere_semaine_epi}** ({derniere_annee})")

# ============================================================
# 5. ENRICHISSEMENT DONNÉES SPATIALES - POPULATION
# ============================================================

@st.cache_data
def worldpop_children_stats(_sa_gdf, use_gee):
    """Extraction population WorldPop (0-14 ans) ou simulation"""
    if not use_gee:
        # Simulation réaliste basée sur la superficie
        pop_base = _sa_gdf.geometry.area.apply(lambda x: max(5000, int(x * 1e6 * np.random.uniform(50, 200))))
        return pd.DataFrame({
            "ADM3_NAME": _sa_gdf["ADM3_NAME"], 
            "Pop_Totale": pop_base,
            "Pop_Moins_15": (pop_base * np.random.uniform(0.42, 0.52, len(_sa_gdf))).astype(int)
        })
    
    try:
        # Conversion GeoDataFrame -> FeatureCollection GEE
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"ADM3_NAME": row["ADM3_NAME"]}
            
            if geom.geom_type == 'Polygon':
                coords = [[[x, y] for x, y in geom.exterior.coords]]
                ee_geom = ee.Geometry.Polygon(coords)
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for poly in geom.geoms:
                    coords.append([[[x, y] for x, y in poly.exterior.coords]])
                ee_geom = ee.Geometry.MultiPolygon(coords)
            else:
                continue
            
            features.append(ee.Feature(ee_geom, props))
        
        fc = ee.FeatureCollection(features)
        
        # WorldPop par âge
        pop = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic()
        
        # Bandes enfants (0-14 ans)
        male_bands = ["M_0", "M_1", "M_5", "M_10"]
        female_bands = ["F_0", "F_1", "F_5", "F_10"]
        
        pop_children = pop.select(male_bands + female_bands)
        
        stats = pop_children.reduceRegions(
            collection=fc, 
            reducer=ee.Reducer.sum(), 
            scale=100
        )
        
        stats_info = stats.getInfo()
        
        data_list = []
        for feat in stats_info['features']:
            props = feat['properties']
            pop_sum = sum([props.get(band, 0) for band in male_bands + female_bands])
            data_list.append({
                "ADM3_NAME": props.get("ADM3_NAME", ""),
                "Pop_Moins_15": int(pop_sum),
                "Pop_Totale": int(pop_sum * 2.2)  # Facteur standard
            })
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ WorldPop indisponible: Simulation activée")
        pop_base = _sa_gdf.geometry.area.apply(lambda x: max(5000, int(x * 1e6 * np.random.uniform(50, 200))))
        return pd.DataFrame({
            "ADM3_NAME": _sa_gdf["ADM3_NAME"], 
            "Pop_Totale": pop_base,
            "Pop_Moins_15": (pop_base * np.random.uniform(0.42, 0.52, len(_sa_gdf))).astype(int)
        })

with st.spinner("🔄 Extraction données population..."):
    pop_df = worldpop_children_stats(sa_gdf, gee_ok)

# ============================================================
# 6. URBANISATION – GHSL SMOD
# ============================================================

@st.cache_data
def urban_classification(_sa_gdf, use_gee):
    """Classification urbain/rural via GHSL ou simulation"""
    if not use_gee:
        # Simulation basée sur la densité
        return pd.DataFrame({
            "ADM3_NAME": _sa_gdf["ADM3_NAME"],
            "Urbanisation": np.random.choice(
                ["Urbain", "Rural", "Semi-urbain"], 
                len(_sa_gdf), 
                p=[0.15, 0.65, 0.20]
            )
        })
    
    try:
        # Conversion GeoDataFrame -> FeatureCollection GEE
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"ADM3_NAME": row["ADM3_NAME"]}
            
            if geom.geom_type == 'Polygon':
                coords = [[[x, y] for x, y in geom.exterior.coords]]
                ee_geom = ee.Geometry.Polygon(coords)
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for poly in geom.geoms:
                    coords.append([[[x, y] for x, y in poly.exterior.coords]])
                ee_geom = ee.Geometry.MultiPolygon(coords)
            else:
                continue
            
            features.append(ee.Feature(ee_geom, props))
        
        fc = ee.FeatureCollection(features)
        smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD/2020")
        
        def classify(feature):
            stats = smod.reduceRegion(
                ee.Reducer.mode(), 
                feature.geometry(), 
                scale=1000, 
                maxPixels=1e9
            )
            smod_value = ee.Number(stats.get("smod_code")).toInt()
            
            urbanisation = ee.Algorithms.If(
                smod_value.gte(30), "Urbain",
                ee.Algorithms.If(smod_value.eq(23), "Semi-urbain", "Rural")
            )
            
            return feature.set({"Urbanisation": urbanisation})
        
        urban_fc = fc.map(classify)
        urban_info = urban_fc.getInfo()
        
        data_list = []
        for feat in urban_info['features']:
            props = feat['properties']
            data_list.append({
                "ADM3_NAME": props.get("ADM3_NAME", ""),
                "Urbanisation": props.get("Urbanisation", "Rural")
            })
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ GHSL indisponible: Simulation activée")
        return pd.DataFrame({
            "ADM3_NAME": _sa_gdf["ADM3_NAME"],
            "Urbanisation": np.random.choice(
                ["Urbain", "Rural", "Semi-urbain"], 
                len(_sa_gdf), 
                p=[0.15, 0.65, 0.20]
            )
        })

with st.spinner("🔄 Classification urbaine..."):
    urban_df = urban_classification(sa_gdf, gee_ok)

# ============================================================
# 7. CLIMAT – NASA POWER
# ============================================================

@st.cache_data(ttl=86400)
def fetch_climate_nasa_power(_sa_gdf, start_date, end_date):
    """Récupération données climatiques NASA POWER"""
    data_list = []
    
    for idx, row in _sa_gdf.iterrows():
        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,RH2M",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON"
        }
        
        try:
            r = requests.get(url, params=params, timeout=30)
            j = r.json()
            
            if "properties" in j and "parameter" in j["properties"]:
                p = j["properties"]["parameter"]
                
                temp_values = list(p.get("T2M", {}).values())
                rh_values = list(p.get("RH2M", {}).values())
                
                temp_mean = np.nanmean(temp_values) if temp_values else 28.0
                rh_mean = np.nanmean(rh_values) if rh_values else 50.0
                
                data_list.append({
                    "ADM3_NAME": row["ADM3_NAME"],
                    "Temperature_Moy": temp_mean,
                    "Humidite_Moy": rh_mean,
                    "Saison_Seche_Humidite": rh_mean * 0.7
                })
            else:
                data_list.append({
                    "ADM3_NAME": row["ADM3_NAME"],
                    "Temperature_Moy": 28.0 + np.random.uniform(-3, 3),
                    "Humidite_Moy": 50.0 + np.random.uniform(-10, 10),
                    "Saison_Seche_Humidite": 35.0 + np.random.uniform(-5, 5)
                })
                
        except:
            data_list.append({
                "ADM3_NAME": row["ADM3_NAME"],
                "Temperature_Moy": 28.0 + np.random.uniform(-3, 3),
                "Humidite_Moy": 50.0 + np.random.uniform(-10, 10),
                "Saison_Seche_Humidite": 35.0 + np.random.uniform(-5, 5)
            })
    
    return pd.DataFrame(data_list)

with st.spinner("🔄 Récupération données climatiques..."):
    climate_df = fetch_climate_nasa_power(sa_gdf, start_date, end_date)

# ============================================================
# FUSION DES DONNÉES
# ============================================================

sa_gdf_enrichi = sa_gdf.copy()
sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df, on="ADM3_NAME", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df, on="ADM3_NAME", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df, on="ADM3_NAME", how="left")

# Calculer superficie et densité
sa_gdf_enrichi["Superficie_km2"] = sa_gdf_enrichi.geometry.area / 1e6
sa_gdf_enrichi["Densite_Pop"] = sa_gdf_enrichi["Pop_Totale"] / sa_gdf_enrichi["Superficie_km2"].replace(0, 1)
sa_gdf_enrichi["Densite_Moins_15"] = sa_gdf_enrichi["Pop_Moins_15"] / sa_gdf_enrichi["Superficie_km2"].replace(0, 1)

# Remplacer NaN/Inf
sa_gdf_enrichi = sa_gdf_enrichi.replace([np.inf, -np.inf], 0)
sa_gdf_enrichi = sa_gdf_enrichi.fillna(0)

st.sidebar.success("✓ Enrichissement spatial terminé")
# ============================================================
# 8. KPIs GLOBAUX
# ============================================================

st.header("📊 Indicateurs Clés de Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📈 Cas totaux", f"{len(df):,}")
with col2:
    taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
    delta_vac = taux_non_vac - 45  # Comparé à l'objectif de 45%
    st.metric("💉 Non vaccinés", f"{taux_non_vac:.1f}%", delta=f"{delta_vac:+.1f}%")
with col3:
    age_median = df["Age_Mois"].median()
    st.metric("👶 Âge médian", f"{int(age_median)} mois")
with col4:
    if "Issue" in df.columns:
        taux_deces = (df["Issue"] == "Décédé").mean() * 100
        st.metric("☠️ Létalité", f"{taux_deces:.2f}%")
    else:
        st.metric("☠️ Létalité", "N/A")
with col5:
    n_aires_touchees = df["Aire_Sante"].nunique()
    pct_aires = (n_aires_touchees / len(sa_gdf)) * 100
    st.metric("🗺️ Aires touchées", f"{n_aires_touchees}/{len(sa_gdf)}", delta=f"{pct_aires:.0f}%")

# ============================================================
# 9. ANALYSE PAR AIRE DE SANTÉ
# ============================================================

cases_by_area = df.groupby("Aire_Sante").agg({
    "ID_Cas": "count",
    "Statut_Vaccinal": lambda x: (x == "Non").mean() * 100,
    "Age_Mois": "mean"
}).reset_index()
cases_by_area.columns = ["Aire_Sante", "Cas_Observes", "Taux_Non_Vaccines", "Age_Moyen"]

# Fusionner avec données spatiales
sa_gdf_with_cases = sa_gdf_enrichi.merge(
    cases_by_area, 
    left_on="ADM3_NAME", 
    right_on="Aire_Sante", 
    how="left"
)
sa_gdf_with_cases["Cas_Observes"] = sa_gdf_with_cases["Cas_Observes"].fillna(0)
sa_gdf_with_cases["Taux_Non_Vaccines"] = sa_gdf_with_cases["Taux_Non_Vaccines"].fillna(0)

# ============================================================
# 10. ANALYSE TEMPORELLE PAR SEMAINES ÉPIDÉMIOLOGIQUES
# ============================================================

st.header("📈 Analyse Temporelle par Semaines Épidémiologiques")

# Agrégation par semaine épidémiologique
weekly_cases = df.groupby(['Annee', 'Semaine_Epi']).size().reset_index(name='Cas')
weekly_cases['Semaine_Label'] = weekly_cases['Annee'].astype(str) + '-S' + weekly_cases['Semaine_Epi'].astype(str).str.zfill(2)

# Graphique courbe épidémique
fig_epi = go.Figure()

fig_epi.add_trace(go.Scatter(
    x=weekly_cases['Semaine_Label'],
    y=weekly_cases['Cas'],
    mode='lines+markers',
    name='Cas observés',
    line=dict(color='#d32f2f', width=3),
    marker=dict(size=6),
    hovertemplate='<b>%{x}</b><br>Cas: %{y}<extra></extra>'
))

# Ligne de tendance
from scipy.signal import savgol_filter
if len(weekly_cases) > 5:
    tendance = savgol_filter(weekly_cases['Cas'], window_length=min(7, len(weekly_cases) if len(weekly_cases) % 2 == 1 else len(weekly_cases)-1), polyorder=2)
    fig_epi.add_trace(go.Scatter(
        x=weekly_cases['Semaine_Label'],
        y=tendance,
        mode='lines',
        name='Tendance',
        line=dict(color='#1976d2', width=2, dash='dash'),
        hovertemplate='<b>%{x}</b><br>Tendance: %{y:.1f}<extra></extra>'
    ))

# Seuil d'alerte
fig_epi.add_hline(
    y=seuil_alerte_epidemique, 
    line_dash="dot", 
    line_color="orange",
    annotation_text=f"Seuil d'alerte ({seuil_alerte_epidemique} cas/sem)",
    annotation_position="right"
)

fig_epi.update_layout(
    title="Courbe épidémique par semaines épidémiologiques",
    xaxis_title="Semaine épidémiologique",
    yaxis_title="Nombre de cas",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_epi, use_container_width=True)

# Statistiques par semaine
col1, col2, col3 = st.columns(3)

with col1:
    semaine_max = weekly_cases.loc[weekly_cases['Cas'].idxmax()]
    st.metric(
        "🔴 Semaine avec pic maximal",
        semaine_max['Semaine_Label'],
        f"{int(semaine_max['Cas'])} cas"
    )

with col2:
    cas_moyen_semaine = weekly_cases['Cas'].mean()
    st.metric(
        "📊 Moyenne hebdomadaire",
        f"{cas_moyen_semaine:.1f} cas"
    )

with col3:
    # Variation dernière semaine
    if len(weekly_cases) >= 2:
        variation = weekly_cases.iloc[-1]['Cas'] - weekly_cases.iloc[-2]['Cas']
        pct_variation = (variation / weekly_cases.iloc[-2]['Cas'] * 100) if weekly_cases.iloc[-2]['Cas'] > 0 else 0
        st.metric(
            "📉 Variation dernière semaine",
            f"{int(variation):+d} cas",
            f"{pct_variation:+.1f}%"
        )

# ============================================================
# 11. ANALYSE PAR TRANCHE D'ÂGE
# ============================================================

st.subheader("👶 Distribution par Tranches d'Âge")

df["Tranche_Age"] = pd.cut(
    df["Age_Mois"],
    bins=[0, 12, 60, 120, 180],
    labels=["0-1 an", "1-5 ans", "5-10 ans", "10-15 ans"]
)

age_stats = df.groupby("Tranche_Age").agg({
    "ID_Cas": "count",
    "Statut_Vaccinal": lambda x: (x == "Non").mean() * 100
}).reset_index()
age_stats.columns = ["Tranche_Age", "Nombre_Cas", "Pct_Non_Vaccines"]

col1, col2 = st.columns(2)

with col1:
    fig_age = px.bar(
        age_stats,
        x="Tranche_Age",
        y="Nombre_Cas",
        title="Cas par tranche d'âge",
        color="Nombre_Cas",
        color_continuous_scale="Reds",
        text="Nombre_Cas"
    )
    fig_age.update_traces(textposition='outside')
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_vacc_age = px.bar(
        age_stats,
        x="Tranche_Age",
        y="Pct_Non_Vaccines",
        title="% non vaccinés par âge",
        color="Pct_Non_Vaccines",
        color_continuous_scale="Oranges",
        text="Pct_Non_Vaccines"
    )
    fig_vacc_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_vacc_age, use_container_width=True)

# ============================================================
# 12. NOWCASTING - CORRECTION DES DÉLAIS
# ============================================================

st.subheader("⏱️ Nowcasting - Correction des Délais de Notification")

df["Delai_Notification"] = (df["Date_Notification"] - df["Date_Debut_Eruption"]).dt.days

delai_moyen = df["Delai_Notification"].mean()
delai_median = df["Delai_Notification"].median()
delai_std = df["Delai_Notification"].std()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Délai moyen", f"{delai_moyen:.1f} jours")
with col2:
    st.metric("Délai médian", f"{delai_median:.0f} jours")
with col3:
    st.metric("Écart-type", f"{delai_std:.1f} jours")
with col4:
    # Correction pour la dernière semaine
    derniere_semaine_label = weekly_cases.iloc[-1]['Semaine_Label']
    cas_derniere_semaine = int(weekly_cases.iloc[-1]['Cas'])
    
    # Facteur de correction (méthode Nobbs & Law)
    facteur_correction = 1 + (delai_moyen / 7)
    cas_corriges = int(cas_derniere_semaine * facteur_correction)
    
    st.metric(
        f"Cas corrigés ({derniere_semaine_label})", 
        cas_corriges, 
        delta=cas_corriges - cas_derniere_semaine
    )

# Distribution des délais
fig_delai = px.histogram(
    df,
    x="Delai_Notification",
    nbins=20,
    title="Distribution des délais de notification",
    labels={"Delai_Notification": "Délai (jours)", "count": "Nombre de cas"},
    color_discrete_sequence=['#d32f2f']
)
fig_delai.add_vline(x=delai_moyen, line_dash="dash", line_color="blue", annotation_text=f"Moyenne: {delai_moyen:.1f}j")
fig_delai.add_vline(x=delai_median, line_dash="dash", line_color="green", annotation_text=f"Médiane: {delai_median:.0f}j")
st.plotly_chart(fig_delai, use_container_width=True)

# ============================================================
# 13. CARTE INTERACTIVE - SITUATION ACTUELLE
# ============================================================

st.header("🗺️ Cartographie de la Situation Actuelle")

center_lat = sa_gdf_with_cases.geometry.centroid.y.mean()
center_lon = sa_gdf_with_cases.geometry.centroid.x.mean()

m = folium.Map(
    location=[center_lat, center_lon], 
    zoom_start=6, 
    tiles="CartoDB positron",
    control_scale=True
)

# Colormap
import branca.colormap as cm
max_cases = sa_gdf_with_cases["Cas_Observes"].max()

if max_cases > 0:
    colormap = cm.LinearColormap(
        colors=['#e8f5e9', '#81c784', '#ffeb3b', '#ff9800', '#f44336', '#b71c1c'],
        vmin=0,
        vmax=max_cases,
        caption="Nombre de cas observés"
    )
    colormap.add_to(m)

# Ajouter les polygones avec popups enrichis
for idx, row in sa_gdf_with_cases.iterrows():
    popup_html = f"""
    <div style="font-family: Arial; width: 350px;">
        <h3 style="margin-bottom: 10px; color: #1976d2; border-bottom: 2px solid #1976d2;">
            {row['ADM3_NAME']}
        </h3>
        
        <div style="background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin: 0; color: #d32f2f;">📊 Situation Épidémiologique</h4>
            <table style="width: 100%; margin-top: 5px;">
                <tr><td><b>Cas observés:</b></td><td style="text-align: right;"><b style="font-size: 18px; color: #d32f2f;">{int(row['Cas_Observes'])}</b></td></tr>
                <tr><td>Non vaccinés:</td><td style="text-align: right;">{row['Taux_Non_Vaccines']:.1f}%</td></tr>
                <tr><td>Âge moyen:</td><td style="text-align: right;">{row.get('Age_Moyen', 0):.0f} mois</td></tr>
            </table>
        </div>
        
        <div style="background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin: 0; color: #1976d2;">👥 Démographie</h4>
            <table style="width: 100%; margin-top: 5px;">
                <tr><td>Population totale:</td><td style="text-align: right;">{int(row['Pop_Totale']):,}</td></tr>
                <tr><td>Enfants &lt;15 ans:</td><td style="text-align: right;">{int(row['Pop_Moins_15']):,}</td></tr>
                <tr><td>Densité:</td><td style="text-align: right;">{row['Densite_Pop']:.1f} hab/km²</td></tr>
                <tr><td>Type:</td><td style="text-align: right;"><b>{row['Urbanisation']}</b></td></tr>
            </table>
        </div>
        
        <div style="background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin: 0; color: #4caf50;">🌡️ Climat</h4>
            <table style="width: 100%; margin-top: 5px;">
                <tr><td>Température moy:</td><td style="text-align: right;">{row['Temperature_Moy']:.1f}°C</td></tr>
                <tr><td>Humidité moy:</td><td style="text-align: right;">{row['Humidite_Moy']:.1f}%</td></tr>
                <tr><td>Humidité (saison sèche):</td><td style="text-align: right;">{row['Saison_Seche_Humidite']:.1f}%</td></tr>
            </table>
        </div>
    </div>
    """
    
    fill_color = colormap(row['Cas_Observes']) if max_cases > 0 else '#e0e0e0'
    
    # Style selon niveau de risque
    if row['Cas_Observes'] >= seuil_alerte_epidemique:
        line_color = '#b71c1c'
        line_weight = 3
    else:
        line_color = 'black'
        line_weight = 1
    
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, color=fill_color, weight=line_weight, border=line_color: {
            'fillColor': color,
            'color': border,
            'weight': weight,
            'fillOpacity': 0.7
        },
        tooltip=folium.Tooltip(
            f"<b>{row['ADM3_NAME']}</b><br>{int(row['Cas_Observes'])} cas<br>Pop: {int(row['Pop_Totale']):,}",
            sticky=True
        ),
        popup=folium.Popup(popup_html, max_width=400)
    ).add_to(m)

# Heatmap des cas
heat_data = [
    [row.geometry.centroid.y, row.geometry.centroid.x, row['Cas_Observes']] 
    for idx, row in sa_gdf_with_cases.iterrows() if row['Cas_Observes'] > 0
]

if heat_data:
    HeatMap(
        heat_data, 
        radius=20, 
        blur=25, 
        max_zoom=13,
        gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'}
    ).add_to(m)

# Ajouter une légende personnalisée
legend_html = f'''
<div style="position: fixed; bottom: 50px; left: 50px; width: 250px; 
     background-color: white; border: 2px solid grey; z-index:9999; font-size:14px;
     padding: 10px; border-radius: 5px;">
     <p style="margin: 0; font-weight: bold;">📊 Légende</p>
     <p style="margin: 5px 0;"><span style="background-color: #e8f5e9; padding: 2px 8px;">Faible</span> 0-{max_cases//3:.0f} cas</p>
     <p style="margin: 5px 0;"><span style="background-color: #ffeb3b; padding: 2px 8px;">Moyen</span> {max_cases//3:.0f}-{2*max_cases//3:.0f} cas</p>
     <p style="margin: 5px 0;"><span style="background-color: #f44336; padding: 2px 8px; color: white;">Élevé</span> >{2*max_cases//3:.0f} cas</p>
     <p style="margin: 5px 0; padding-top: 5px; border-top: 1px solid #ccc;">
         <b>Seuil alerte:</b> {seuil_alerte_epidemique} cas/sem
     </p>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=1400, height=650)

# Statistiques cartographiques
col1, col2, col3 = st.columns(3)

with col1:
    aires_alerte = len(sa_gdf_with_cases[sa_gdf_with_cases['Cas_Observes'] >= seuil_alerte_epidemique])
    st.metric("🚨 Aires en alerte", aires_alerte, f"{aires_alerte/len(sa_gdf)*100:.1f}%")

with col2:
    aires_sans_cas = len(sa_gdf_with_cases[sa_gdf_with_cases['Cas_Observes'] == 0])
    st.metric("✅ Aires sans cas", aires_sans_cas, f"{aires_sans_cas/len(sa_gdf)*100:.1f}%")

with col3:
    densite_cas_moy = (sa_gdf_with_cases['Cas_Observes'] / sa_gdf_with_cases['Superficie_km2']).mean()
    st.metric("📍 Densité moyenne", f"{densite_cas_moy:.2f} cas/km²")
# ============================================================
# PARTIE 4/4 - MODÉLISATION PRÉDICTIVE PAR SEMAINES ÉPIDÉMIOLOGIQUES
# ============================================================

st.header("🔮 Modélisation Prédictive par Semaines Épidémiologiques")

st.markdown(f"""
<div class="info-box">
<b>Configuration de la prédiction:</b><br>
• Dernière semaine de données: <b>S{derniere_semaine_epi} ({derniere_annee})</b><br>
• Période de prédiction: <b>{pred_mois} mois ({n_weeks_pred} semaines)</b><br>
• Semaines prédites: <b>S{derniere_semaine_epi+1} à S{min(derniere_semaine_epi+n_weeks_pred, 52)}</b><br>
• Seuils configurés: Baisse ≥{seuil_baisse}%, Hausse ≥{seuil_hausse}%
</div>
""", unsafe_allow_html=True)

# Bouton pour lancer la prédiction
if 'prediction_lancee' not in st.session_state:
    st.session_state.prediction_lancee = False

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("🚀 Lancer la Modélisation Prédictive", type="primary", use_container_width=True):
        st.session_state.prediction_lancee = True
        st.rerun()
with col2:
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.prediction_lancee = False
        st.rerun()

if st.session_state.prediction_lancee:
    
    with st.spinner("🤖 Entraînement du modèle et génération des prédictions..."):
        
        # ============================================================
        # PRÉPARATION DES DONNÉES PAR SEMAINE ÉPIDÉMIOLOGIQUE
        # ============================================================
        
        # Agrégation par aire et semaine épidémiologique
        weekly_features = df.groupby(["Aire_Sante", "Annee", "Semaine_Epi"]).agg(
            Cas_Observes=("ID_Cas", "count"),
            Non_Vaccines=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100),
            Age_Moyen=("Age_Mois", "mean")
        ).reset_index()
        
        # Créer label de semaine
        weekly_features['Semaine_Label'] = (
            weekly_features['Annee'].astype(str) + '-S' + 
            weekly_features['Semaine_Epi'].astype(str).str.zfill(2)
        )
        
        # Fusionner avec données spatiales
        weekly_features = weekly_features.merge(
            sa_gdf_enrichi[[
                "ADM3_NAME", "Pop_Totale", "Pop_Moins_15", "Densite_Moins_15", 
                "Urbanisation", "Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite"
            ]],
            left_on="Aire_Sante",
            right_on="ADM3_NAME",
            how="left"
        )
        
        # Encoder variables catégorielles
        le_urban = LabelEncoder()
        weekly_features["Urban_Encoded"] = le_urban.fit_transform(
            weekly_features["Urbanisation"].fillna("Rural")
        )
        
        # Features lag (historique 1-4 semaines)
        weekly_features = weekly_features.sort_values(['Aire_Sante', 'Annee', 'Semaine_Epi'])
        for lag in [1, 2, 3, 4]:
            weekly_features[f'Cas_Lag_{lag}'] = weekly_features.groupby('Aire_Sante')['Cas_Observes'].shift(lag)
        
        # Remplacer NaN
        numeric_cols = weekly_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            weekly_features[col] = weekly_features[col].replace([np.inf, -np.inf], 0)
            weekly_features[col] = weekly_features[col].fillna(weekly_features[col].mean() if weekly_features[col].mean() == weekly_features[col].mean() else 0)
        
        # ============================================================
        # ENTRAÎNEMENT DU MODÈLE AMÉLIORÉ
        # ============================================================
        
        st.subheader("📚 Entraînement du Modèle")
        
        # Features pour le modèle
        feature_cols = [
            "Cas_Observes", "Non_Vaccines", "Age_Moyen", "Pop_Totale", "Pop_Moins_15", 
            "Densite_Moins_15", "Urban_Encoded", "Temperature_Moy", "Saison_Seche_Humidite",
            "Cas_Lag_1", "Cas_Lag_2", "Cas_Lag_3", "Cas_Lag_4", "Semaine_Epi"
        ]
        
        # Filtrer les données avec historique complet
        weekly_features_clean = weekly_features.dropna(subset=feature_cols)
        
        if len(weekly_features_clean) < 20:
            st.warning("⚠️ Données insuffisantes pour l'entraînement (minimum 20 observations avec historique)")
            st.info("💡 Ajoutez plus de données historiques ou réduisez la période d'analyse")
            st.stop()
        
        X = weekly_features_clean[feature_cols]
        y = weekly_features_clean["Cas_Observes"]
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Entraîner plusieurs modèles et choisir le meilleur
        models = {
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300, 
                learning_rate=0.05, 
                max_depth=5, 
                min_samples_split=4,
                random_state=42
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                random_state=42
            )
        }
        
        best_model = None
        best_score = -np.inf
        best_name = ""
        
        col1, col2 = st.columns(2)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            
            if cv_mean > best_score:
                best_score = cv_mean
                best_model = model
                best_name = name
            
            with col1 if name == "GradientBoosting" else col2:
                st.metric(
                    f"📊 {name}",
                    f"R² = {score:.3f}",
                    f"CV: {cv_mean:.3f} (±{cv_scores.std():.3f})"
                )
        
        st.success(f"✓ Meilleur modèle sélectionné: **{best_name}** (R² CV = {best_score:.3f})")
        
        # Importance des variables
        feature_importance = pd.DataFrame({
            "Variable": feature_cols,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        with st.expander("📊 Importance des variables", expanded=False):
            fig_imp = px.bar(
                feature_importance.head(10),
                x="Importance",
                y="Variable",
                orientation="h",
                title="Top 10 variables les plus importantes",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        # ============================================================
        # GÉNÉRATION DES PRÉDICTIONS PAR SEMAINE ÉPIDÉMIOLOGIQUE
        # ============================================================
        
        st.subheader(f"📅 Prédictions - {n_weeks_pred} Semaines Épidémiologiques")
        
        future_predictions = []
        
        # Pour chaque aire de santé
        for aire in weekly_features["Aire_Sante"].unique():
            aire_data = weekly_features[weekly_features["Aire_Sante"] == aire].sort_values(['Annee', 'Semaine_Epi'])
            
            if aire_data.empty:
                continue
            
            # Dernière observation
            last_obs = aire_data.iloc[-1]
            
            # Historique des 4 dernières semaines
            last_4_weeks = aire_data.tail(4)['Cas_Observes'].values
            if len(last_4_weeks) < 4:
                last_4_weeks = np.pad(last_4_weeks, (4-len(last_4_weeks), 0), 'edge')
            
            # Prédire semaine par semaine
            for i in range(1, n_weeks_pred + 1):
                # Calculer la nouvelle semaine épidémiologique
                nouvelle_semaine_epi = (derniere_semaine_epi + i - 1) % 52 + 1
                nouvelle_annee = derniere_annee + ((derniere_semaine_epi + i - 1) // 52)
                
                # Préparer les features
                future_row = {
                    "Aire_Sante": aire,
                    "Annee": nouvelle_annee,
                    "Semaine_Epi": nouvelle_semaine_epi,
                    "Semaine_Label": f"{nouvelle_annee}-S{str(nouvelle_semaine_epi).zfill(2)}",
                    "Non_Vaccines": last_obs["Non_Vaccines"],
                    "Age_Moyen": last_obs["Age_Moyen"],
                    "Pop_Totale": last_obs["Pop_Totale"],
                    "Pop_Moins_15": last_obs["Pop_Moins_15"],
                    "Densite_Moins_15": last_obs["Densite_Moins_15"],
                    "Urban_Encoded": last_obs["Urban_Encoded"],
                    "Temperature_Moy": last_obs["Temperature_Moy"],
                    "Saison_Seche_Humidite": last_obs["Saison_Seche_Humidite"]
                }
                
                # Pour la première prédiction, utiliser les cas observés
                if i == 1:
                    future_row["Cas_Observes"] = last_obs["Cas_Observes"]
                    future_row["Cas_Lag_1"] = last_4_weeks[-1]
                    future_row["Cas_Lag_2"] = last_4_weeks[-2] if len(last_4_weeks) >= 2 else last_4_weeks[-1]
                    future_row["Cas_Lag_3"] = last_4_weeks[-3] if len(last_4_weeks) >= 3 else last_4_weeks[-1]
                    future_row["Cas_Lag_4"] = last_4_weeks[-4] if len(last_4_weeks) >= 4 else last_4_weeks[-1]
                else:
                    # Utiliser les prédictions précédentes
                    prev_predictions = [p["Predicted_Cases"] for p in future_predictions if p["Aire_Sante"] == aire]
                    
                    future_row["Cas_Observes"] = prev_predictions[-1] if prev_predictions else last_obs["Cas_Observes"]
                    future_row["Cas_Lag_1"] = prev_predictions[-1] if len(prev_predictions) >= 1 else last_4_weeks[-1]
                    future_row["Cas_Lag_2"] = prev_predictions[-2] if len(prev_predictions) >= 2 else last_4_weeks[-2]
                    future_row["Cas_Lag_3"] = prev_predictions[-3] if len(prev_predictions) >= 3 else last_4_weeks[-3]
                    future_row["Cas_Lag_4"] = prev_predictions[-4] if len(prev_predictions) >= 4 else last_4_weeks[-4]
                
                # Prédire
                X_future = np.array([[
                    future_row[col] for col in feature_cols
                ]])
                X_future_scaled = scaler.transform(X_future)
                predicted_cases = max(0, best_model.predict(X_future_scaled)[0])
                
                future_row["Predicted_Cases"] = predicted_cases
                future_predictions.append(future_row)
        
        # Convertir en DataFrame
        future_df = pd.DataFrame(future_predictions)
        
        st.success(f"✓ {len(future_df)} prédictions générées ({len(future_df['Aire_Sante'].unique())} aires × {n_weeks_pred} semaines)")
        
        # ============================================================
        # ANALYSE DES PRÉDICTIONS
        # ============================================================
        
        # Calculer la moyenne historique par aire
        moyenne_historique = weekly_features.groupby("Aire_Sante")["Cas_Observes"].mean().reset_index()
        moyenne_historique.columns = ["Aire_Sante", "Moyenne_Historique"]
        
        # Statistiques par aire
        risk_df = future_df.groupby("Aire_Sante").agg(
            Cas_Predits_Total=("Predicted_Cases", "sum"),
            Cas_Predits_Max=("Predicted_Cases", "max"),
            Cas_Predits_Moyen=("Predicted_Cases", "mean"),
            Semaine_Pic=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(), "Semaine_Label"] if len(x) > 0 else "N/A")
        ).reset_index()
        
        # Fusionner avec moyenne historique
        risk_df = risk_df.merge(moyenne_historique, on="Aire_Sante", how="left")
        
        # Calculer variation par rapport à la moyenne
        risk_df["Variation_Pct"] = ((risk_df["Cas_Predits_Moyen"] - risk_df["Moyenne_Historique"]) / 
                                     risk_df["Moyenne_Historique"].replace(0, 1)) * 100
        
        # Catégoriser
        risk_df["Categorie_Variation"] = pd.cut(
            risk_df["Variation_Pct"],
            bins=[-np.inf, -seuil_baisse, -10, 10, seuil_hausse, np.inf],
            labels=["Forte baisse", "Baisse modérée", "Stable", "Hausse modérée", "Forte hausse"]
        )
        
        # Trier par variation
        risk_df = risk_df.sort_values("Variation_Pct", ascending=False)
        
        # ============================================================
        # VISUALISATIONS DES PRÉDICTIONS
        # ============================================================
        
        st.subheader("📊 Synthèse des Prédictions")
        
        # KPIs prédictions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predits = risk_df["Cas_Predits_Total"].sum()
            st.metric("Total cas prédits", f"{int(total_predits):,}")
        
        with col2:
            aires_hausse = len(risk_df[risk_df["Variation_Pct"] >= seuil_hausse])
            st.metric("🔴 Aires en hausse", aires_hausse, f"≥{seuil_hausse}%")
        
        with col3:
            aires_baisse = len(risk_df[risk_df["Variation_Pct"] <= -seuil_baisse])
            st.metric("🟢 Aires en baisse", aires_baisse, f"≥{seuil_baisse}%")
        
        with col4:
            aires_stables = len(risk_df[(risk_df["Variation_Pct"] > -10) & (risk_df["Variation_Pct"] < 10)])
            st.metric("⚪ Aires stables", aires_stables, "±10%")
        
        # Distribution des variations
        fig_var_dist = px.pie(
            risk_df,
            names="Categorie_Variation",
            title="Distribution des tendances prédites",
            color="Categorie_Variation",
            color_discrete_map={
                "Forte baisse": "#2e7d32",
                "Baisse modérée": "#81c784",
                "Stable": "#9e9e9e",
                "Hausse modérée": "#ff9800",
                "Forte hausse": "#d32f2f"
            }
        )
        st.plotly_chart(fig_var_dist, use_container_width=True)
        
        # ============================================================
        # HEATMAP TEMPORELLE DES PRÉDICTIONS
        # ============================================================
        
        st.subheader("🔥 Heatmap Temporelle des Prédictions")
        
        # Préparer données pour heatmap
        heatmap_data = future_df.pivot(
            index="Aire_Sante",
            columns="Semaine_Label",
            values="Predicted_Cases"
        ).fillna(0)
        
        # Top 20 aires avec variation la plus forte (hausse ou baisse)
        top_10_hausse = risk_df.head(10)["Aire_Sante"].tolist()
        top_10_baisse = risk_df.tail(10)["Aire_Sante"].tolist()
        top_20_aires = top_10_hausse + top_10_baisse
        heatmap_data_top = heatmap_data.loc[heatmap_data.index.isin(top_20_aires)]
        
        # Créer heatmap avec gradient vert clair -> rouge foncé
        fig_heatmap = px.imshow(
            heatmap_data_top,
            labels=dict(x="Semaine Épidémiologique", y="Aire de Santé", color="Cas prédits"),
            x=heatmap_data_top.columns,
            y=heatmap_data_top.index,
            title=f"Évolution prédite - Top 20 aires (10 hausses + 10 baisses)",
            color_continuous_scale=["#e8f5e9", "#81c784", "#ffeb3b", "#ff9800", "#f44336", "#b71c1c"],
            aspect="auto"
        )
        fig_heatmap.update_xaxes(side="bottom", tickangle=-45)
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # ============================================================
        # TABLEAUX AVEC SEUILS
        # ============================================================
        
        tab1, tab2, tab3 = st.tabs([
            f"🔴 Hausse ≥{seuil_hausse}%",
            f"🟢 Baisse ≥{seuil_baisse}%",
            "📊 Toutes les aires"
        ])
        
        with tab1:
            st.subheader(f"Aires avec Hausse Significative (≥{seuil_hausse}%)")
            
            hausse_df = risk_df[risk_df["Variation_Pct"] >= seuil_hausse].copy()
            
            if len(hausse_df) > 0:
                def highlight_hausse(row):
                    return ["background-color: #ffcdd2"] * len(row)
                
                st.dataframe(
                    hausse_df[[
                        "Aire_Sante", "Moyenne_Historique", "Cas_Predits_Moyen",
                        "Variation_Pct", "Semaine_Pic", "Cas_Predits_Max"
                    ]].style.apply(highlight_hausse, axis=1).format({
                        "Moyenne_Historique": "{:.1f}",
                        "Cas_Predits_Moyen": "{:.1f}",
                        "Variation_Pct": "{:+.1f}%",
                        "Cas_Predits_Max": "{:.0f}"
                    }),
                    use_container_width=True
                )
                
                st.warning(f"⚠️ **{len(hausse_df)} aire(s)** nécessite(nt) une vigilance accrue")
            else:
                st.success("✓ Aucune aire avec hausse significative")
        
        with tab2:
            st.subheader(f"Aires avec Baisse Significative (≥{seuil_baisse}%)")
            
            baisse_df = risk_df[risk_df["Variation_Pct"] <= -seuil_baisse].copy()
            
            if len(baisse_df) > 0:
                def highlight_baisse(row):
                    return ["background-color: #c8e6c9"] * len(row)
                
                st.dataframe(
                    baisse_df[[
                        "Aire_Sante", "Moyenne_Historique", "Cas_Predits_Moyen",
                        "Variation_Pct", "Semaine_Pic", "Cas_Predits_Max"
                    ]].style.apply(highlight_baisse, axis=1).format({
                        "Moyenne_Historique": "{:.1f}",
                        "Cas_Predits_Moyen": "{:.1f}",
                        "Variation_Pct": "{:+.1f}%",
                        "Cas_Predits_Max": "{:.0f}"
                    }),
                    use_container_width=True
                )
                
                st.success(f"✓ **{len(baisse_df)} aire(s)** montre(nt) une amélioration")
            else:
                st.info("ℹ️ Aucune aire avec baisse significative")
        
        with tab3:
            st.subheader("Tableau Complet des Prédictions")
            
            def highlight_all(row):
                if row["Variation_Pct"] >= seuil_hausse:
                    return ["background-color: #ffcdd2"] * len(row)
                elif row["Variation_Pct"] <= -seuil_baisse:
                    return ["background-color: #c8e6c9"] * len(row)
                else:
                    return [""] * len(row)
            
            st.dataframe(
                risk_df[[
                    "Aire_Sante", "Moyenne_Historique", "Cas_Predits_Moyen",
                    "Variation_Pct", "Categorie_Variation", "Semaine_Pic", "Cas_Predits_Max"
                ]].style.apply(highlight_all, axis=1).format({
                    "Moyenne_Historique": "{:.1f}",
                    "Cas_Predits_Moyen": "{:.1f}",
                    "Variation_Pct": "{:+.1f}%",
                    "Cas_Predits_Max": "{:.0f}"
                }),
                use_container_width=True,
                height=400
            )
        
        # ============================================================
        # CARTE DES PRÉDICTIONS
        # ============================================================
        
        st.subheader("🗺️ Carte des Prédictions")
        
        sa_gdf_pred = sa_gdf_enrichi.merge(
            risk_df,
            left_on="ADM3_NAME",
            right_on="Aire_Sante",
            how="left"
        )
        sa_gdf_pred["Variation_Pct"] = sa_gdf_pred["Variation_Pct"].fillna(0)
        
        m_pred = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=6, 
            tiles="CartoDB positron"
        )
        
        # Colormap basée sur la variation
        max_var = max(abs(sa_gdf_pred["Variation_Pct"].min()), abs(sa_gdf_pred["Variation_Pct"].max()))
        
        colormap_pred = cm.LinearColormap(
            colors=['#2e7d32', '#81c784', '#e0e0e0', '#ff9800', '#d32f2f'],
            vmin=-max_var,
            vmax=max_var,
            caption="Variation (%) par rapport à la moyenne"
        )
        colormap_pred.add_to(m_pred)
        
        for idx, row in sa_gdf_pred.iterrows():
            popup_html = f"""
            <div style="font-family: Arial; width: 360px;">
                <h3 style="color: #1976d2; border-bottom: 2px solid #1976d2;">
                    {row['ADM3_NAME']}
                </h3>
                
                <div style="background-color: {'#ffebee' if row['Variation_Pct'] >= seuil_hausse else '#e8f5e9' if row['Variation_Pct'] <= -seuil_baisse else '#f5f5f5'}; 
                            padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <h4 style="margin: 0;">🔮 Prédictions</h4>
                    <table style="width: 100%; margin-top: 5px;">
                        <tr><td><b>Moyenne historique:</b></td><td style="text-align: right;">{row.get('Moyenne_Historique', 0):.1f} cas/sem</td></tr>
                        <tr><td><b>Moyenne prédite:</b></td><td style="text-align: right;">{row.get('Cas_Predits_Moyen', 0):.1f} cas/sem</td></tr>
                        <tr><td><b>Variation:</b></td><td style="text-align: right; font-size: 18px; color: {'#d32f2f' if row['Variation_Pct'] >= seuil_hausse else '#2e7d32' if row['Variation_Pct'] <= -seuil_baisse else '#000'};">
                            <b>{row.get('Variation_Pct', 0):+.1f}%</b>
                        </td></tr>
                        <tr><td>Tendance:</td><td style="text-align: right;"><b>{row.get('Categorie_Variation', 'N/A')}</b></td></tr>
                        <tr><td>Semaine du pic:</td><td style="text-align: right;">{row.get('Semaine_Pic', 'N/A')}</td></tr>
                        <tr><td>Pic maximal:</td><td style="text-align: right;">{int(row.get('Cas_Predits_Max', 0))} cas</td></tr>
                    </table>
                </div>
            </div>
            """
            
            fill_color = colormap_pred(row['Variation_Pct']) if pd.notna(row['Variation_Pct']) else '#e0e0e0'
            
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, color=fill_color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                tooltip=folium.Tooltip(
                    f"<b>{row['ADM3_NAME']}</b><br>Variation: {row.get('Variation_Pct', 0):+.1f}%",
                    sticky=True
                ),
                popup=folium.Popup(popup_html, max_width=400)
            ).add_to(m_pred)
        
        st_folium(m_pred, width=1400, height=650)
        
        # ============================================================
        # EXPORT DES RÉSULTATS
        # ============================================================
        
        st.header("💾 Export des Résultats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_pred = risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Synthèse prédictions (CSV)",
                data=csv_pred,
                file_name=f"predictions_synthese_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_detail = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📅 Détail par semaine (CSV)",
                data=csv_detail,
                file_name=f"predictions_detail_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            geojson_str = sa_gdf_pred.to_json()
            st.download_button(
                label="🗺️ Carte prédictions (GeoJSON)",
                data=geojson_str,
                file_name=f"carte_predictions_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/json"
            )
        
        # Export Excel complet
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            risk_df.to_excel(writer, sheet_name='Synthese', index=False)
            future_df.to_excel(writer, sheet_name='Detail_Semaines', index=False)
            cases_by_area.to_excel(writer, sheet_name='Cas_Observes', index=False)
            age_stats.to_excel(writer, sheet_name='Analyse_Age', index=False)
            weekly_cases.to_excel(writer, sheet_name='Historique_Hebdo', index=False)
        
        st.download_button(
            label="📊 Rapport complet (Excel)",
            data=output.getvalue(),
            file_name=f"rapport_complet_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        # ============================================================
        # RECOMMANDATIONS FINALES
        # ============================================================
        
        st.header("💡 Recommandations Opérationnelles")
        
        aires_critiques_hausse = risk_df[risk_df["Variation_Pct"] >= seuil_hausse]["Aire_Sante"].tolist()
        aires_amelioration = risk_df[risk_df["Variation_Pct"] <= -seuil_baisse]["Aire_Sante"].tolist()
        
        if aires_critiques_hausse:
            st.error(f"🚨 **{len(aires_critiques_hausse)} aire(s) à risque CRITIQUE (hausse ≥{seuil_hausse}%)**")
            
            for i, aire in enumerate(aires_critiques_hausse[:5], 1):
                aire_info = risk_df[risk_df["Aire_Sante"] == aire].iloc[0]
                st.write(f"{i}. **{aire}**: {aire_info['Variation_Pct']:+.1f}% - Pic S{aire_info['Semaine_Pic']} ({int(aire_info['Cas_Predits_Max'])} cas)")
            
            if len(aires_critiques_hausse) > 5:
                st.caption(f"... et {len(aires_critiques_hausse)-5} autre(s)")
            
            st.write("")
            st.write("**Actions prioritaires recommandées:**")
            st.write("✅ Renforcer la surveillance épidémiologique hebdomadaire dans ces aires")
            st.write("✅ Organiser des campagnes de vaccination de rattrapage urgentes")
            st.write("✅ Prépositionner les stocks de vaccins et intrants médicaux")
            st.write("✅ Sensibiliser les communautés aux signes d'alerte de la rougeole")
            st.write("✅ Coordonner avec les partenaires (OMS, UNICEF, MSF)")
            st.write("✅ Préparer les structures de santé à une augmentation de cas")
        
        if aires_amelioration:
            st.success(f"✓ **{len(aires_amelioration)} aire(s) montrent une amélioration (baisse ≥{seuil_baisse}%)**")
            
            st.write("**Bonnes pratiques à capitaliser:**")
            st.write("• Documenter les interventions ayant conduit à cette baisse")
            st.write("• Partager les leçons apprises avec les autres aires")
            st.write("• Maintenir la vigilance pour éviter une résurgence")
        
        if not aires_critiques_hausse and not aires_amelioration:
            st.info("ℹ️ **Situation stable** - Aucune variation significative détectée")
            st.write("• Maintenir la surveillance de routine")
            st.write("• Continuer les activités de vaccination selon le calendrier")
        
        st.markdown("---")
        st.caption(f"""
        **Méthodologie de prédiction:**  
        Modèle: {best_name} | Score R² (validation croisée): {best_score:.3f}  
        Variables: Historique (4 semaines), démographie, urbanisation, climat, vaccination  
        Période de prédiction: S{derniere_semaine_epi+1} à S{min(derniere_semaine_epi+n_weeks_pred, 52)} ({n_weeks_pred} semaines)  
        Seuils: Baisse ≥{seuil_baisse}%, Hausse ≥{seuil_hausse}%, Alerte épidémique ≥{seuil_alerte_epidemique} cas/sem
        """)

else:
    st.info("👆 Cliquez sur le bouton ci-dessus pour lancer la modélisation prédictive")
    
    st.markdown("""
    ### 📚 Ce que vous obtiendrez :
    
    ✅ **Prédictions par semaines épidémiologiques** (S1 à S52)  
    ✅ **Identification des aires à risque** selon vos seuils personnalisés  
    ✅ **Heatmap temporelle** (évolution semaine par semaine)  
    ✅ **Cartes interactives** avec prédictions  
    ✅ **Export multi-formats** (CSV, Excel, GeoJSON)  
    ✅ **Recommandations opérationnelles** basées sur les résultats
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(f"""
Dashboard de Surveillance Rougeole - Version 2.0 (Semaines Épidémiologiques)  
Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} | Pays: {pays_selectionne} ({iso3_pays}) | Période: {start_date} - {end_date}  
Nombre d'aires: {len(sa_gdf)} | Cas analysés: {len(df):,}
""")
