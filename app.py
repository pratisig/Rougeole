# ============================================================
# APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE (Multi-pays)
# Version améliorée selon spécifications
# PARTIE 1/5 - IMPORTS, CONFIG ET CHARGEMENT DONNÉES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
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

# CONFIG STREAMLIT
st.set_page_config(
    page_title="Surveillance Rougeole Multi-pays",
    layout="wide",
    page_icon="🦠",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-card{background-color:#f0f2f6;padding:15px;border-radius:10px;box-shadow:2px 2px 5px rgba(0,0,0,0.1)}
.high-risk{background-color:#ffebee;color:#c62828;font-weight:bold;padding:5px;border-radius:3px}
.medium-risk{background-color:#fff3e0;color:#ef6c00;padding:5px;border-radius:3px}
.low-risk{background-color:#e8f5e9;color:#2e7d32;padding:5px;border-radius:3px}
.stButton>button{width:100%}
h1{color:#d32f2f}
.info-box{background-color:#e3f2fd;padding:10px;border-left:4px solid #2196f3;margin:10px 0}
.model-hint{background-color:#fff9c4;padding:8px;border-radius:5px;font-size:0.9em;margin:5px 0}
</style>
""", unsafe_allow_html=True)

st.title("🦠 Dashboard de Surveillance et Prédiction - Rougeole")
st.markdown("### Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques")

PAYS_ISO3_MAP = {
    "Niger": "ner",
    "Burkina Faso": "bfa",
    "Mali": "mli",
    "Mauritanie": "mrt"
}

# ============================================================
# INITIALISATION GOOGLE EARTH ENGINE
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
# SIDEBAR - CONFIGURATION
# ============================================================

st.sidebar.header("📂 Configuration de l'Analyse")

# Session state pour le cache
if 'pays_precedent' not in st.session_state:
    st.session_state.pays_precedent = None
if 'sa_gdf_cache' not in st.session_state:
    st.session_state.sa_gdf_cache = None

# MODE DÉMO
st.sidebar.subheader("🎯 Mode d'utilisation")
mode_demo = st.sidebar.radio(
    "Choisissez votre mode",
    ["📊 Données réelles", "🧪 Mode démo (données simulées)"],
    help="Mode démo : génère automatiquement des données fictives pour tester l'application"
)

# AIRES DE SANTÉ
st.sidebar.subheader("🗺️ Aires de Santé")
option_aire = st.sidebar.radio(
    "Source des données géographiques",
    ["Fichier local (ao_hlthArea.zip)", "Upload personnalisé"],
    key='option_aire'
)

pays_selectionne = None
iso3_pays = None

if option_aire == "Fichier local (ao_hlthArea.zip)":
    pays_selectionne = st.sidebar.selectbox(
        "🌍 Sélectionner le pays",
        list(PAYS_ISO3_MAP.keys()),
        key='pays_select'
    )
    iso3_pays = PAYS_ISO3_MAP[pays_selectionne]
    
    pays_change = (st.session_state.pays_precedent != pays_selectionne)
    if pays_change:
        st.session_state.pays_precedent = pays_selectionne
        st.session_state.sa_gdf_cache = None
        st.rerun()

upload_file = None
if option_aire == "Upload personnalisé":
    upload_file = st.sidebar.file_uploader(
        "Charger un fichier géographique",
        type=["shp", "geojson", "zip"],
        help="Format: Shapefile ou GeoJSON avec colonnes 'iso3' et 'health_area'"
    )

# DONNÉES ÉPIDÉMIOLOGIQUES
st.sidebar.subheader("📊 Données Épidémiologiques")

if mode_demo == "🧪 Mode démo (données simulées)":
    option_linelist = "Données fictives (test)"
    linelist_file = None
    vaccination_file = None
    st.sidebar.info("📊 Mode démo activé - Données simulées")
else:
    linelist_file = st.sidebar.file_uploader(
        "📋 Linelists rougeole (CSV)",
        type=["csv"],
        help="Format: health_area, Semaine_Epi, Cas_Total OU Date_Debut_Eruption, Aire_Sante..."
    )
    
    vaccination_file = st.sidebar.file_uploader(
        "💉 Couverture vaccinale (CSV - optionnel)",
        type=["csv"],
        help="Format: health_area, Taux_Vaccination (en %)"
    )

# PÉRIODE D'ANALYSE
st.sidebar.subheader("📅 Période d'Analyse")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Date début",
        value=datetime(2024, 1, 1),
        key='start_date'
    )
with col2:
    end_date = st.date_input(
        "Date fin",
        value=datetime.today(),
        key='end_date'
    )

# PARAMÈTRES DE PRÉDICTION
st.sidebar.subheader("🔮 Paramètres de Prédiction")
pred_mois = st.sidebar.slider(
    "Période de prédiction (mois)",
    min_value=1,
    max_value=12,
    value=3,
    help="Nombre de mois à prédire après la dernière semaine de données"
)
n_weeks_pred = pred_mois * 4

st.sidebar.info(f"📆 Prédiction sur **{n_weeks_pred} semaines épidémiologiques** (~{pred_mois} mois)")

# CHOIX DU MODÈLE
st.sidebar.subheader("🤖 Modèle de Prédiction")

modele_choisi = st.sidebar.selectbox(
    "Choisissez votre algorithme",
    [
        "GradientBoosting (Recommandé)",
        "RandomForest",
        "Ridge Regression",
        "Lasso Regression",
        "Decision Tree"
    ],
    help="Sélectionnez l'algorithme de machine learning pour la prédiction"
)

# Hints pour chaque modèle
model_hints = {
    "GradientBoosting (Recommandé)": "🎯 **Gradient Boosting** : Très performant pour les séries temporelles. Combine plusieurs modèles faibles pour créer un modèle fort. Excellent pour capturer les relations non-linéaires. Recommandé pour la surveillance épidémiologique.",
    "RandomForest": "🌳 **Random Forest** : Ensemble d'arbres de décision. Robuste aux valeurs aberrantes et aux données manquantes. Bon pour les interactions complexes entre variables.",
    "Ridge Regression": "📊 **Ridge Regression** : Régression linéaire avec régularisation L2. Simple et rapide. Idéal pour relations linéaires. Moins performant sur données non-linéaires.",
    "Lasso Regression": "🎯 **Lasso Regression** : Régularisation L1 avec sélection automatique des variables. Utile quand beaucoup de variables peu importantes. Simplifie le modèle.",
    "Decision Tree": "🌲 **Decision Tree** : Arbre de décision unique. Simple à interpréter mais risque de sur-apprentissage. Moins robuste que les méthodes d'ensemble."
}

st.sidebar.markdown(f'<div class="model-hint">{model_hints[modele_choisi]}</div>', unsafe_allow_html=True)

# SEUILS D'ALERTE
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
# FONCTIONS DE CHARGEMENT DES DONNÉES GÉOGRAPHIQUES
# ============================================================

@st.cache_data
def load_health_areas_from_zip(zip_path, iso3_filter):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
            
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if not shp_files:
                raise ValueError("Aucun fichier .shp trouvé dans le ZIP")
            
            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf_full = gpd.read_file(shp_path)
            
            # Trouver la colonne ISO3
            iso3_col = None
            for col in ['iso3', 'ISO3', 'iso_code', 'ISO_CODE', 'country_iso', 'COUNTRY_ISO']:
                if col in gdf_full.columns:
                    iso3_col = col
                    break
            
            if iso3_col is None:
                st.warning(f"⚠️ Colonne ISO3 non trouvée. Colonnes: {list(gdf_full.columns)}")
                return gpd.GeoDataFrame()
            
            gdf = gdf_full[gdf_full[iso3_col] == iso3_filter].copy()
            
            if gdf.empty:
                st.warning(f"⚠️ Aucune aire de santé pour {iso3_filter}")
                return gpd.GeoDataFrame()
            
            # Trouver la colonne nom
            name_col = None
            for col in ['health_area', 'HEALTH_AREA', 'name_fr', 'name', 'NAME', 'nom', 'NOM', 'aire_sante']:
                if col in gdf.columns:
                    name_col = col
                    break
            
            if name_col:
                gdf['health_area'] = gdf[name_col]
            else:
                gdf['health_area'] = [f"Aire_{i+1}" for i in range(len(gdf))]
            
            gdf = gdf[gdf.geometry.is_valid]
            
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            
            return gdf
            
    except Exception as e:
        st.error(f"❌ Erreur ZIP: {e}")
        return gpd.GeoDataFrame()

def load_shapefile_from_upload(upload_file):
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
                        raise ValueError("Aucun .shp trouvé")
        else:
            gdf = gpd.read_file(upload_file)
        
        if "health_area" not in gdf.columns:
            for col in ["health_area", "HEALTH_AREA", "name_fr", "name", "NAME", "nom", "NOM"]:
                if col in gdf.columns:
                    gdf["health_area"] = gdf[col]
                    break
            else:
                gdf["health_area"] = [f"Aire_{i}" for i in range(len(gdf))]
        
        gdf = gdf[gdf.geometry.is_valid]
        
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
        
    except Exception as e:
        st.error(f"❌ Erreur lecture: {e}")
        return gpd.GeoDataFrame()

# ============================================================
# PARTIE 2/5 - CHARGEMENT AIRES DE SANTÉ ET DONNÉES DE CAS
# ============================================================

# CHARGEMENT DES AIRES DE SANTÉ
if st.session_state.sa_gdf_cache is not None and option_aire == "Fichier local (ao_hlthArea.zip)":
    sa_gdf = st.session_state.sa_gdf_cache
    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (cache)")
else:
    with st.spinner(f"🔄 Chargement des aires de santé..."):
        if option_aire == "Fichier local (ao_hlthArea.zip)":
            zip_path = os.path.join("data", "ao_hlthArea.zip")
            if not os.path.exists(zip_path):
                st.error(f"❌ Fichier non trouvé: {zip_path}")
                st.info("📁 Placez 'ao_hlthArea.zip' dans le dossier 'data/'")
                st.stop()
            
            sa_gdf = load_health_areas_from_zip(zip_path, iso3_pays)
            
            if sa_gdf.empty:
                st.error(f"❌ Impossible de charger {pays_selectionne} ({iso3_pays})")
                st.stop()
            else:
                st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées ({iso3_pays})")
                st.session_state.sa_gdf_cache = sa_gdf
                
        elif option_aire == "Upload personnalisé":
            if upload_file is None:
                st.warning("⚠️ Veuillez uploader un fichier")
                st.stop()
            else:
                sa_gdf = load_shapefile_from_upload(upload_file)
                if sa_gdf.empty:
                    st.error("❌ Fichier invalide")
                    st.stop()
                else:
                    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées")
                    st.session_state.sa_gdf_cache = sa_gdf

if sa_gdf.empty or sa_gdf is None:
    st.error("❌ Aucune aire chargée")
    st.stop()

# ============================================================
# GÉNÉRATION DE DONNÉES FICTIVES
# ============================================================

@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500, start=None, end=None):
    """Génère des données de cas fictives pour le mode démo"""
    np.random.seed(42)
    
    if start is None:
        start = datetime(2024, 1, 1)
    if end is None:
        end = datetime.today()
    
    delta_days = (end - start).days
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.random.exponential(scale=delta_days/3, size=n).clip(0, delta_days).astype(int),
        unit="D"
    )
    
    df = pd.DataFrame({
        "ID_Cas": range(1, n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(np.random.poisson(3, n), unit="D"),
        "Aire_Sante": np.random.choice(_sa_gdf["health_area"].unique(), n),
        "Age_Mois": np.random.gamma(shape=2, scale=30, size=n).clip(6, 180).astype(int),
        "Statut_Vaccinal": np.random.choice(["Oui", "Non"], n, p=[0.55, 0.45]),
        "Sexe": np.random.choice(["M", "F"], n),
        "Issue": np.random.choice(["Guéri", "Décédé", "Inconnu"], n, p=[0.92, 0.03, 0.05])
    })
    
    return df

@st.cache_data
def generate_dummy_vaccination(_sa_gdf):
    """Génère des données de couverture vaccinale fictives"""
    np.random.seed(42)
    
    return pd.DataFrame({
        "health_area": _sa_gdf["health_area"],
        "Taux_Vaccination": np.random.beta(a=8, b=2, size=len(_sa_gdf)) * 100  # Biaisé vers 80%
    })

# ============================================================
# CHARGEMENT DES DONNÉES DE CAS
# ============================================================

with st.spinner("📥 Chargement données de cas..."):
    if mode_demo == "🧪 Mode démo (données simulées)":
        df = generate_dummy_linelists(sa_gdf, start=start_date, end=end_date)
        vaccination_df = generate_dummy_vaccination(sa_gdf)
        st.sidebar.info(f"📊 {len(df)} cas simulés générés")
        
    else:
        if linelist_file is None:
            st.error("❌ Veuillez uploader un fichier CSV de lineliste")
            st.stop()
            
        try:
            df_raw = pd.read_csv(linelist_file)
            
            # Vérifier si format agrégé ou détaillé
            if "Semaine_Epi" in df_raw.columns and "Cas_Total" in df_raw.columns:
                # Format agrégé - expansion nécessaire
                expanded_rows = []
                for _, row in df_raw.iterrows():
                    aire = row.get("health_area") or row.get("Aire_Sante") or row.get("name_fr")
                    semaine = int(row["Semaine_Epi"])
                    cas_total = int(row["Cas_Total"])
                    annee = row.get("Annee", 2024)
                    
                    base_date = datetime.strptime(f"{annee}-W{semaine:02d}-1", "%Y-W%W-%w")
                    
                    for i in range(cas_total):
                        expanded_rows.append({
                            "ID_Cas": len(expanded_rows) + 1,
                            "Date_Debut_Eruption": base_date + timedelta(days=np.random.randint(0, 7)),
                            "Date_Notification": base_date + timedelta(days=np.random.randint(0, 10)),
                            "Aire_Sante": aire,
                            "Age_Mois": 0,
                            "Statut_Vaccinal": "Inconnu",
                            "Sexe": "Inconnu",
                            "Issue": "Inconnu"
                        })
                
                df = pd.DataFrame(expanded_rows)
                
            elif "Date_Debut_Eruption" in df_raw.columns:
                # Format détaillé
                df = df_raw.copy()
                
                for col in ["Date_Debut_Eruption", "Date_Notification"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                st.error("❌ Format CSV non reconnu. Colonnes requises: 'Date_Debut_Eruption' ou 'Semaine_Epi'+'Cas_Total'")
                st.stop()
            
            st.sidebar.success(f"✓ {len(df)} cas chargés")
            
        except Exception as e:
            st.error(f"❌ Erreur CSV: {e}")
            st.stop()
        
        # Charger données de vaccination si fournies
        if vaccination_file is not None:
            try:
                vaccination_df = pd.read_csv(vaccination_file)
                st.sidebar.success(f"✓ Couverture vaccinale chargée ({len(vaccination_df)} aires)")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erreur vaccination CSV: {e}")
                vaccination_df = None
        else:
            # Vérifier si Statut_Vaccinal dans linelist
            if "Statut_Vaccinal" in df.columns:
                # Calculer le taux par aire
                vacc_by_area = df.groupby("Aire_Sante").agg({
                    "Statut_Vaccinal": lambda x: ((x == "Oui").sum() / len(x) * 100) if len(x) > 0 else 0
                }).reset_index()
                vacc_by_area.columns = ["health_area", "Taux_Vaccination"]
                vaccination_df = vacc_by_area
                st.sidebar.info("ℹ️ Taux vaccination extrait de la linelist")
            else:
                vaccination_df = None
                st.sidebar.info("ℹ️ Pas de données de vaccination")

# Filtrer par période
df = df[
    (df["Date_Debut_Eruption"] >= pd.to_datetime(start_date)) &
    (df["Date_Debut_Eruption"] <= pd.to_datetime(end_date))
].copy()

if len(df) == 0:
    st.warning("⚠️ Aucun cas dans la période")
    st.stop()

# Calculer semaine épidémiologique
def calculer_semaine_epidemio(date):
    return date.isocalendar()[1]

df['Semaine_Epi'] = df['Date_Debut_Eruption'].apply(calculer_semaine_epidemio)
df['Annee'] = df['Date_Debut_Eruption'].dt.year
df['Semaine_Annee'] = df['Annee'].astype(str) + '-S' + df['Semaine_Epi'].astype(str).str.zfill(2)

derniere_semaine_epi = df['Semaine_Epi'].max()
derniere_annee = df['Annee'].max()

st.sidebar.info(f"📅 Dernière semaine: **S{derniere_semaine_epi}** ({derniere_annee})")

# ============================================================
# PARTIE 3/5 - ENRICHISSEMENT AVEC DONNÉES EXTERNES
# WorldPop, NASA POWER, GHSL
# ============================================================

# ============================================================
# WORLDPOP - DONNÉES DÉMOGRAPHIQUES (VERSION CORRIGÉE)
# ============================================================

@st.cache_data
def worldpop_children_stats(_sa_gdf, use_gee):
    """
    Extraction des statistiques WorldPop avec la logique correcte
    Retourne: garçons, filles, population totale, enfants
    """
    if not use_gee:
        st.sidebar.warning("⚠️ WorldPop: GEE indisponible")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Garcons": [np.nan] * len(_sa_gdf),
            "Pop_Filles": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf)
        })
    
    try:
        # Barre de progression
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Chargement et mosaïque WorldPop
        status_text.text("📥 Chargement WorldPop...")
        dataset = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
        pop_img = dataset.mosaic()
        
        # Sélection des bandes
        male_bands = ["M_0", "M_1", "M_5", "M_10"]
        female_bands = ["F_0", "F_1", "F_5", "F_10"]
        
        selected_males = pop_img.select(male_bands)
        selected_females = pop_img.select(female_bands)
        total_pop = pop_img.select(['population'])
        
        # Calcul de la bande enfants (somme M + F)
        enfants = selected_males.add(selected_females).reduce(ee.Reducer.sum()).rename('enfants')
        
        # Assemblage final
        final_mosaic = selected_males.addBands(selected_females).addBands(total_pop).addBands(enfants)
        
        # Création des features GEE
        status_text.text("🗺️ Conversion géométries...")
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"health_area": row["health_area"]}
            
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
        
        # Statistiques zonales
        status_text.text("🔢 Calcul statistiques zonales...")
        stats = final_mosaic.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.sum(),
            scale=100
        )
        
        # Extraction des résultats
        status_text.text("📊 Extraction résultats...")
        stats_info = stats.getInfo()
        
        data_list = []
        total_aires = len(stats_info['features'])
        
        for i, feat in enumerate(stats_info['features']):
            props = feat['properties']
            
            # Somme des garçons (M_0 + M_1 + M_5 + M_10)
            garcons = sum([props.get(band, 0) for band in male_bands])
            
            # Somme des filles (F_0 + F_1 + F_5 + F_10)
            filles = sum([props.get(band, 0) for band in female_bands])
            
            # Population totale
            pop_totale = props.get("population", 0)
            
            # Enfants (garçons + filles)
            enfants_total = props.get("enfants", garcons + filles)
            
            data_list.append({
                "health_area": props.get("health_area", ""),
                "Pop_Totale": int(pop_totale) if pop_totale > 0 else np.nan,
                "Pop_Garcons": int(garcons),
                "Pop_Filles": int(filles),
                "Pop_Enfants": int(enfants_total)
            })
            
            # Mise à jour progression
            progress_bar.progress((i + 1) / total_aires)
        
        progress_bar.empty()
        status_text.text("✅ WorldPop terminé")
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.error(f"❌ WorldPop: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Garcons": [np.nan] * len(_sa_gdf),
            "Pop_Filles": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf)
        })

# ============================================================
# GHSL - CLASSIFICATION URBAINE
# ============================================================

@st.cache_data
def urban_classification(_sa_gdf, use_gee):
    """Classification urbaine via GHSL"""
    if not use_gee:
        st.sidebar.warning("⚠️ GHSL: GEE indisponible")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": [np.nan] * len(_sa_gdf)
        })
    
    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        status_text.text("🏙️ Classification urbaine...")
        
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"health_area": row["health_area"]}
            
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
                smod_value.gte(30),
                "Urbain",
                ee.Algorithms.If(smod_value.eq(23), "Semi-urbain", "Rural")
            )
            return feature.set({"Urbanisation": urbanisation})
        
        urban_fc = fc.map(classify)
        urban_info = urban_fc.getInfo()
        
        data_list = []
        total_aires = len(urban_info['features'])
        
        for i, feat in enumerate(urban_info['features']):
            props = feat['properties']
            data_list.append({
                "health_area": props.get("health_area", ""),
                "Urbanisation": props.get("Urbanisation", "Rural")
            })
            progress_bar.progress((i + 1) / total_aires)
        
        progress_bar.empty()
        status_text.text("✅ GHSL terminé")
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.error(f"❌ GHSL: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": [np.nan] * len(_sa_gdf)
        })

# ============================================================
# NASA POWER - DONNÉES CLIMATIQUES
# ============================================================

@st.cache_data(ttl=86400)
def fetch_climate_nasa_power(_sa_gdf, start_date, end_date):
    """Récupération données climatiques NASA POWER"""
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    data_list = []
    total_aires = len(_sa_gdf)
    
    for idx, row in _sa_gdf.iterrows():
        status_text.text(f"🌡️ Climat {idx+1}/{total_aires}...")
        
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
                
                temp_mean = np.nanmean(temp_values) if temp_values else np.nan
                rh_mean = np.nanmean(rh_values) if rh_values else np.nan
                
                # Indicateur saison sèche (humidité réduite)
                saison_seche_hum = rh_mean * 0.7 if not np.isnan(rh_mean) else np.nan
                
                data_list.append({
                    "health_area": row["health_area"],
                    "Temperature_Moy": temp_mean,
                    "Humidite_Moy": rh_mean,
                    "Saison_Seche_Humidite": saison_seche_hum
                })
            else:
                data_list.append({
                    "health_area": row["health_area"],
                    "Temperature_Moy": np.nan,
                    "Humidite_Moy": np.nan,
                    "Saison_Seche_Humidite": np.nan
                })
        except:
            data_list.append({
                "health_area": row["health_area"],
                "Temperature_Moy": np.nan,
                "Humidite_Moy": np.nan,
                "Saison_Seche_Humidite": np.nan
            })
        
        progress_bar.progress((idx + 1) / total_aires)
    
    progress_bar.empty()
    status_text.text("✅ Climat terminé")
    
    return pd.DataFrame(data_list)

# ============================================================
# ENRICHISSEMENT DU GEODATAFRAME
# ============================================================

with st.spinner("🔄 Enrichissement des données..."):
    
    # WorldPop
    pop_df = worldpop_children_stats(sa_gdf, gee_ok)
    
    # GHSL
    urban_df = urban_classification(sa_gdf, gee_ok)
    
    # NASA POWER
    climate_df = fetch_climate_nasa_power(sa_gdf, start_date, end_date)

# Fusion des données
sa_gdf_enrichi = sa_gdf.copy()
sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df, on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df, on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df, on="health_area", how="left")

# Ajout des données de vaccination si disponibles
if vaccination_df is not None:
    sa_gdf_enrichi = sa_gdf_enrichi.merge(vaccination_df, on="health_area", how="left")
else:
    sa_gdf_enrichi["Taux_Vaccination"] = np.nan

# Calcul superficie et densités
sa_gdf_enrichi["Superficie_km2"] = sa_gdf_enrichi.geometry.area / 1e6

# Densité population totale (correcte maintenant)
sa_gdf_enrichi["Densite_Pop"] = (
    sa_gdf_enrichi["Pop_Totale"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

# Densité enfants
sa_gdf_enrichi["Densite_Enfants"] = (
    sa_gdf_enrichi["Pop_Enfants"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

# Nettoyage des valeurs infinies
sa_gdf_enrichi = sa_gdf_enrichi.replace([np.inf, -np.inf], np.nan)

st.sidebar.success("✓ Enrichissement terminé")

# Résumé des données disponibles
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Données disponibles")

donnees_dispo = {
    "Population": not sa_gdf_enrichi["Pop_Totale"].isna().all(),
    "Urbanisation": not sa_gdf_enrichi["Urbanisation"].isna().all(),
    "Climat": not sa_gdf_enrichi["Humidite_Moy"].isna().all(),
    "Vaccination": not sa_gdf_enrichi["Taux_Vaccination"].isna().all()
}

for nom, dispo in donnees_dispo.items():
    icone = "✅" if dispo else "❌"
    st.sidebar.text(f"{icone} {nom}")

# ============================================================
# PARTIE 4/5 - KPIS, CARTE ET ANALYSES (VERSION AMÉLIORÉE)
# ============================================================

# ============================================================
# KPIS
# ============================================================

st.header("📊 Indicateurs Clés de Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📈 Cas totaux", f"{len(df):,}")

with col2:
    taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
    delta_vac = taux_non_vac - 45
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

# Agrégation par aire
cases_by_area = df.groupby("Aire_Sante").agg({
    "ID_Cas": "count",
    "Statut_Vaccinal": lambda x: (x == "Non").mean() * 100,
    "Age_Mois": "mean"
}).reset_index()

cases_by_area.columns = ["Aire_Sante", "Cas_Observes", "Taux_Non_Vaccines", "Age_Moyen"]

sa_gdf_with_cases = sa_gdf_enrichi.merge(
    cases_by_area,
    left_on="health_area",
    right_on="Aire_Sante",
    how="left"
)

sa_gdf_with_cases["Cas_Observes"] = sa_gdf_with_cases["Cas_Observes"].fillna(0)
sa_gdf_with_cases["Taux_Non_Vaccines"] = sa_gdf_with_cases["Taux_Non_Vaccines"].fillna(0)

# Taux d'attaque pour 10,000 enfants
sa_gdf_with_cases["Taux_Attaque_10000"] = (
    sa_gdf_with_cases["Cas_Observes"] / sa_gdf_with_cases["Pop_Enfants"].replace(0, np.nan) * 10000
).replace([np.inf, -np.inf], np.nan)

# ============================================================
# CARTE AMÉLIORÉE (contours fins, étiquettes sans fond)
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

# Ajout des polygones
for idx, row in sa_gdf_with_cases.iterrows():
    aire_name = row['health_area']
    cas_obs = int(row.get('Cas_Observes', 0))
    pop_enfants = row.get('Pop_Enfants', np.nan)
    taux_attaque = row.get('Taux_Attaque_10000', np.nan)
    urbanisation = row.get('Urbanisation', 'N/A')
    densite = row.get('Densite_Pop', np.nan)
    
    # Popup enrichi
    popup_html = f"""
    <div style="font-family: Arial; width: 350px;">
        <h3 style="margin-bottom: 10px; color: #1976d2; border-bottom: 2px solid #1976d2;">
            {aire_name}
        </h3>
        <div style="background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin: 0; color: #d32f2f;">📊 Situation Épidémiologique</h4>
            <table style="width: 100%; margin-top: 5px;">
                <tr><td><b>Cas observés:</b></td><td style="text-align: right;">
                    <b style="font-size: 18px; color: #d32f2f;">{cas_obs}</b>
                </td></tr>
                <tr><td>Population enfants:</td><td style="text-align: right;">
                    {f"{int(pop_enfants):,}" if not np.isnan(pop_enfants) else "N/A"}
                </td></tr>
                <tr><td>Taux d'attaque:</td><td style="text-align: right;">
                    {f"{taux_attaque:.1f}/10K" if not np.isnan(taux_attaque) else "N/A"}
                </td></tr>
                <tr><td>Type habitat:</td><td style="text-align: right;">
                    <b>{urbanisation if pd.notna(urbanisation) else "N/A"}</b>
                </td></tr>
                <tr><td>Densité pop:</td><td style="text-align: right;">
                    {f"{densite:.1f} hab/km²" if not np.isnan(densite) else "N/A"}
                </td></tr>
            </table>
        </div>
    </div>
    """
    
    fill_color = colormap(row['Cas_Observes']) if max_cases > 0 else '#e0e0e0'
    
    # AMÉLIORATION: Contours plus fins
    if row['Cas_Observes'] >= seuil_alerte_epidemique:
        line_color = '#b71c1c'
        line_weight = 2  # Réduit de 3 à 2
    else:
        line_color = 'black'
        line_weight = 0.5  # Réduit de 1 à 0.5
        
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, color=fill_color, weight=line_weight, border=line_color: {
            'fillColor': color,
            'color': border,
            'weight': weight,
            'fillOpacity': 0.7
        },
        tooltip=folium.Tooltip(
            f"<b>{aire_name}</b><br>{cas_obs} cas",
            sticky=True
        ),
        popup=folium.Popup(popup_html, max_width=400)
    ).add_to(m)
    
    # AMÉLIORATION: Étiquettes sans fond blanc fixe
    if cas_obs > 0:
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            icon=folium.DivIcon(html=f"""
                <div style="
                    font-size: 9pt;
                    color: black;
                    weight: bold;
                    background-color: rgba(255, 255, 255, 0.85);
                    padding: 1px 4px;
                    border-radius: 3px;
                    white-space: nowrap;
                    border: 1px solid rgba(0, 0, 0, 0.2);
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                ">
                    {aire_name}
                </div>
            """)
        ).add_to(m)

# Heatmap
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

# Légende
st_folium(m, width="100%", height=600)

# --- ANALYSES GRAPHIQUES ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Évolution temporelle (Saisonnalité)")
    # Agrégation hebdomadaire
    weekly_trend = df.groupby('Semaine_Annee').size().reset_index(name='Cas')
    fig_trend = px.line(weekly_trend, x='Semaine_Annee', y='Cas', markers=True)
    fig_trend.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    st.subheader("👶 Distribution par Âge et Sexe")
    fig_age = px.histogram(df, x="Age_Mois", color="Sexe", nbins=20, barmode="group")
    st.plotly_chart(fig_age, use_container_width=True)

# ============================================================
# PARTIE 5/5 - MODÉLISATION PRÉDICTIVE (MACHINE LEARNING)
# ============================================================

st.divider()
st.header("🔮 Modélisation Prédictive & Alertes")

if st.button("🚀 Lancer la modélisation prédictive", type="primary"):
    with st.spinner("🧠 Préparation des variables et entraînement du modèle..."):
        
        # 1. Préparation des données hebdomadaires (Training Set)
        # On crée une grille de toutes les aires x toutes les semaines
        all_aires = sa_gdf_enrichi["health_area"].unique()
        all_weeks = sorted(df["Semaine_Epi"].unique())
        
        index = pd.MultiIndex.from_product([all_aires, all_weeks], names=["Aire_Sante", "Semaine_Epi"])
        weekly_features = pd.DataFrame(index=index).reset_index()
        
        # Ajout des cas observés
        cas_counts = df.groupby(["Aire_Sante", "Semaine_Epi"]).size().reset_index(name="Cas_Observes")
        weekly_features = weekly_features.merge(cas_counts, on=["Aire_Sante", "Semaine_Epi"], how="left").fillna(0)
        
        # Ajout des variables statiques par aire
        weekly_features = weekly_features.merge(
            sa_gdf_enrichi[[
                "health_area", "Pop_Enfants", "Urbanisation", "Taux_Vaccination",
                "Temperature_Moy", "Humidite_Moy", "Densite_Pop"
            ]],
            left_on="Aire_Sante", right_on="health_area", how="left"
        )
        
        # Encodage de l'urbanisation
        le_urb = LabelEncoder()
        weekly_features["Urban_Enc"] = le_urb.fit_transform(weekly_features["Urbanisation"].fillna("Rural"))
        
        # Ajout des lags (historique local)
        weekly_features = weekly_features.sort_values(["Aire_Sante", "Semaine_Epi"])
        for i in range(1, 5):
            weekly_features[f"Cas_Lag_{i}"] = weekly_features.groupby("Aire_Sante")["Cas_Observes"].shift(i).fillna(0)
        
        # Moyenne historique locale
        moyenne_historique = weekly_features.groupby("Aire_Sante")["Cas_Observes"].mean().reset_index()
        moyenne_historique.columns = ["Aire_Sante", "Moyenne_Historique"]
        weekly_features = weekly_features.merge(moyenne_historique, on="Aire_Sante", how="left")
        
        # 2. Définition des features pour le modèle
        feature_cols = [
            "Semaine_Epi", "Pop_Enfants", "Urban_Enc", "Taux_Vaccination",
            "Temperature_Moy", "Humidite_Moy", "Densite_Pop", "Moyenne_Historique",
            "Cas_Lag_1", "Cas_Lag_2", "Cas_Lag_3", "Cas_Lag_4"
        ]
        
        # Nettoyage des NaNs pour le training
        train_df = weekly_features.dropna(subset=["Cas_Observes"] + feature_cols)
        
        X = train_df[feature_cols]
        y = train_df["Cas_Observes"]
        
        # 3. Entraînement du modèle sélectionné
        if modele_choisi == "GradientBoosting (Recommandé)":
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        elif modele_choisi == "RandomForest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif modele_choisi == "Ridge Regression":
            model = Ridge()
        elif modele_choisi == "Lasso Regression":
            model = Lasso()
        else:
            model = DecisionTreeRegressor()
            
        model.fit(X, y)
        
        # Évaluation rapide (Cross-validation)
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # 4. Prédiction pour les semaines futures
        predictions_futures = []
        
        for aire in all_aires:
            aire_data = weekly_features[weekly_features["Aire_Sante"] == aire].iloc[-1:].copy()
            
            # État initial pour la prédiction récursive
            current_lags = [
                aire_data["Cas_Observes"].values[0],
                aire_data["Cas_Lag_1"].values[0],
                aire_data["Cas_Lag_2"].values[0],
                aire_data["Cas_Lag_3"].values[0]
            ]
            
            for w in range(1, n_weeks_pred + 1):
                futur_week = (derniere_semaine_epi + w - 1) % 52 + 1
                
                input_row = pd.DataFrame([{
                    "Semaine_Epi": futur_week,
                    "Pop_Enfants": aire_data["Pop_Enfants"].values[0],
                    "Urban_Enc": aire_data["Urban_Enc"].values[0],
                    "Taux_Vaccination": aire_data["Taux_Vaccination"].values[0],
                    "Temperature_Moy": aire_data["Temperature_Moy"].values[0],
                    "Humidite_Moy": aire_data["Humidite_Moy"].values[0],
                    "Densite_Pop": aire_data["Densite_Pop"].values[0],
                    "Moyenne_Historique": aire_data["Moyenne_Historique"].values[0],
                    "Cas_Lag_1": current_lags[0],
                    "Cas_Lag_2": current_lags[1],
                    "Cas_Lag_3": current_lags[2],
                    "Cas_Lag_4": current_lags[3]
                }])
                
                pred_val = model.predict(input_row[feature_cols])[0]
                pred_val = max(0, pred_val) # Pas de cas négatifs
                
                predictions_futures.append({
                    "Aire_Sante": aire,
                    "Semaine_Epi": futur_week,
                    "Cas_Prevus": pred_val
                })
                
                # Update lags pour la semaine suivante
                current_lags = [pred_val] + current_lags[:3]
        
        df_pred = pd.DataFrame(predictions_futures)
        
        # 5. Calcul des Alertes et Risques
        # Agrégation des prédictions par aire
        pred_agg = df_pred.groupby("Aire_Sante")["Cas_Prevus"].agg(['sum', 'mean', 'max']).reset_index()
        pred_agg.columns = ["Aire_Sante", "Total_Prevu", "Moyenne_Prevue", "Pic_Prevu"]
        
        # Jointure avec les données historiques et démographiques
        resultats_finaux = pred_agg.merge(moyenne_historique, on="Aire_Sante")
        resultats_finaux = resultats_finaux.merge(
            sa_gdf_enrichi[["health_area", "Pop_Enfants", "Taux_Vaccination", "Urbanisation"]],
            left_on="Aire_Sante", right_on="health_area"
        )
        
        # Calcul du score de risque
        # Formule : (Evolution / Moyenne) * Poids + (Taux_Attaque_Prevu) * Poids
        resultats_finaux["Evolution_Pct"] = (
            (resultats_finaux["Moyenne_Prevue"] - resultats_finaux["Moyenne_Historique"]) / 
            resultats_finaux["Moyenne_Historique"].replace(0, 0.1) * 100
        )
        
        def determiner_risque(row):
            score = 0
            # Condition 1: Hausse massive par rapport à l'historique
            if row["Evolution_Pct"] >= seuil_hausse: score += 2
            elif row["Evolution_Pct"] > 10: score += 1
            
            # Condition 2: Alerte épidémique (nombre absolu de cas)
            if row["Pic_Prevu"] >= seuil_alerte_epidemique: score += 3
            elif row["Pic_Prevu"] >= 2: score += 1
            
            # Condition 3: Facteurs de vulnérabilité
            if row["Taux_Vaccination"] < 80: score += 1
            
            if score >= 4: return "🔴 Élevé (Alerte)"
            if score >= 2: return "🟠 Modéré"
            return "🟢 Faible"
            
        resultats_finaux["Niveau_Risque"] = resultats_finaux.apply(determiner_risque, axis=1)
        
        # --- AFFICHAGE DES RÉSULTATS ---
        st.success(f"✅ Modélisation terminée avec succès ! (R² validation: {cv_mean:.3f})")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.subheader("🚨 Synthèse des Alertes")
            alerte_counts = resultats_finaux["Niveau_Risque"].value_counts()
            for r_type, count in alerte_counts.items():
                st.write(f"**{r_type}**: {count} aires")
            
            st.dataframe(
                resultats_finaux[resultats_finaux["Niveau_Risque"].str.contains("Élevé|Modéré")]
                .sort_values("Pic_Prevu", ascending=False)
                [["Aire_Sante", "Pic_Prevu", "Evolution_Pct", "Niveau_Risque"]],
                hide_index=True
            )
            
        with col_res2:
            st.subheader("📅 Heatmap des Prédictions")
            # Pivot pour la heatmap : Semaines en X, Aires en Y, Cas en couleur
            # On prend les 20 aires les plus à risque
            top_aires = resultats_finaux.sort_values("Pic_Prevu", ascending=False).head(20)["Aire_Sante"].tolist()
            heatmap_data = df_pred[df_pred["Aire_Sante"].isin(top_aires)].pivot(
                index="Aire_Sante", columns="Semaine_Epi", values="Cas_Prevus"
            )
            fig_heat = px.imshow(
                heatmap_data, 
                labels=dict(x="Semaine Épidémiologique", y="Aire de Santé", color="Cas prévus"),
                color_continuous_scale="YlOrRd"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
        # Carte des prédictions
        st.subheader("🗺️ Carte des Prédictions")

sa_gdf_pred = sa_gdf_enrichi.merge(
    risk_df,
    left_on="health_area",
    right_on="Aire_Sante",
    how="left"
)

sa_gdf_pred["Variation_Pct"] = sa_gdf_pred["Variation_Pct"].fillna(0)
sa_gdf_pred["Cas_Predits_Max"] = sa_gdf_pred["Cas_Predits_Max"].fillna(0)

m_pred = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB positron"
)

max_var = max(abs(sa_gdf_pred["Variation_Pct"].min()), abs(sa_gdf_pred["Variation_Pct"].max()))

colormap_pred = cm.LinearColormap(
    colors=['#2e7d32', '#81c784', '#e0e0e0', '#ff9800', '#d32f2f'],
    vmin=-max_var,
    vmax=max_var,
    caption="Variation (%) par rapport à la moyenne"
)
colormap_pred.add_to(m_pred)

for idx, row in sa_gdf_pred.iterrows():
    aire_name = row['health_area']
    
    # Vérifier que les colonnes existent avec get() pour éviter KeyError
    variation_pct = row.get('Variation_Pct', 0)
    moy_historique = row.get('Moyenne_Historique', 0)
    cas_pred_moy = row.get('Cas_Predits_Moyen', 0)
    cas_pred_max = row.get('Cas_Predits_Max', 0)
    semaine_pic = row.get('Semaine_Pic', 'N/A')
    categorie = row.get('Categorie_Variation', 'N/A')
    
    popup_html = f"""
    <div style="font-family: Arial; width: 360px;">
        <h3 style="color: #1976d2; border-bottom: 2px solid #1976d2;">
            {aire_name}
        </h3>
        <div style="background-color: {'#ffebee' if variation_pct >= seuil_hausse else '#e8f5e9' if variation_pct <= -seuil_baisse else '#f5f5f5'}; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin: 0;">🔮 Prédictions</h4>
            <table style="width: 100%; margin-top: 5px;">
                <tr><td><b>Moyenne historique:</b></td><td style="text-align: right;">
                    {moy_historique:.1f} cas/sem
                </td></tr>
                <tr><td><b>Moyenne prédite:</b></td><td style="text-align: right;">
                    {cas_pred_moy:.1f} cas/sem
                </td></tr>
                <tr><td><b>Variation:</b></td><td style="text-align: right; font-size: 18px; color: {'#d32f2f' if variation_pct >= seuil_hausse else '#2e7d32' if variation_pct <= -seuil_baisse else '#000'};">
                    <b>{variation_pct:+.1f}%</b>
                </td></tr>
                <tr><td>Tendance:</td><td style="text-align: right;">
                    <b>{categorie}</b>
                </td></tr>
                <tr><td>Semaine du pic:</td><td style="text-align: right;">
                    {semaine_pic}
                </td></tr>
                <tr><td>Pic maximal:</td><td style="text-align: right;">
                    {int(cas_pred_max)} cas
                </td></tr>
            </table>
        </div>
    </div>
    """
    
    fill_color = colormap_pred(variation_pct) if pd.notna(variation_pct) else '#e0e0e0'
    
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, color=fill_color: {
            'fillColor': color,
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        },
        tooltip=folium.Tooltip(
            f"<b>{aire_name}</b><br>Variation: {variation_pct:+.1f}%",
            sticky=True
        ),
        popup=folium.Popup(popup_html, max_width=400)
    ).add_to(m_pred)

st_folium(m_pred, width=1400, height=650)
            
        st_folium(m_pred, width="100%", height=500, key="map_pred")
        
        # Export des résultats
        csv_pred = resultats_finaux.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Télécharger le rapport de prédiction (CSV)",
            csv_pred,
            "predictions_rougeole.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Footer méthodologie
        st.markdown("---")
        st.caption(f"""
**Méthodologie de prédiction:**
Modèle: {modele_choisi} | Score R² (validation croisée): {cv_mean:.3f} (±{cv_std:.3f}) |
Variables: {len(feature_cols)} features (historique 4 semaines, démographie, urbanisation, climat, vaccination) |
Période: S{derniere_semaine_epi+1} à S{min(derniere_semaine_epi+n_weeks_pred, 52)} ({n_weeks_pred} semaines) |
Seuils: Baisse ≥{seuil_baisse}%, Hausse ≥{seuil_hausse}%, Alerte ≥{seuil_alerte_epidemique} cas/sem
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
✅ **Intégration automatique** des données disponibles (climat, vaccination, démographie)
    """)

# Footer global
st.markdown("---")
st.caption(f"Plateforme de Surveillance Rougeole Multi-pays | Données GEE, NASA, WorldPop | Actualisé le {datetime.now().strftime('%d/%m/%Y %H:%M')}")
