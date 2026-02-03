# ============================================================
# APP SURVEILLANCE & PRÉDICTION ROUGEOLE - VERSION 3.0
# PARTIE 1/6 - IMPORTS ET CONFIGURATION
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
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

# Configuration Streamlit
st.set_page_config(
    page_title="Surveillance Rougeole Multi-pays",
    layout="wide",
    page_icon="🦠",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
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
.weight-box{background-color:#e8f5e9;padding:10px;border-radius:5px;margin:10px 0;border-left:4px solid #4caf50}
</style>
""", unsafe_allow_html=True)

st.title("🦠 Dashboard de Surveillance et Prédiction - Rougeole")
st.markdown("### Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques")

# Mapping pays ISO3
PAYS_ISO3_MAP = {
    "Niger": "ner",
    "Burkina Faso": "bfa",
    "Mali": "mli",
    "Mauritanie": "mrt"
}

# Initialisation Google Earth Engine
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

# Session state
if 'pays_precedent' not in st.session_state:
    st.session_state.pays_precedent = None
if 'sa_gdf_cache' not in st.session_state:
    st.session_state.sa_gdf_cache = None

# Configuration Sidebar
st.sidebar.header("📂 Configuration de l'Analyse")

# Mode démo
st.sidebar.subheader("🎯 Mode d'utilisation")
mode_demo = st.sidebar.radio(
    "Choisissez votre mode",
    ["📊 Données réelles", "🧪 Mode démo (données simulées)"],
    help="Mode démo : génère automatiquement des données fictives pour tester l'application"
)

# Aires de santé
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
        help="Format : Shapefile ou GeoJSON avec colonnes 'iso3' et 'health_area'"
    )

# Données épidémiologiques
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
        help="Format : health_area, Semaine_Epi, Cas_Total OU Date_Debut_Eruption, Aire_Sante..."
    )
    
    vaccination_file = st.sidebar.file_uploader(
        "💉 Couverture vaccinale (CSV - optionnel)",
        type=["csv"],
        help="Format : health_area, Taux_Vaccination (en %)"
    )

# Période d'analyse
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

# Paramètres de prédiction
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

# Choix du modèle
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

# ========== SYSTÈME HYBRIDE D'IMPORTANCE DES VARIABLES ==========
st.sidebar.subheader("⚖️ Importance des Variables")

mode_importance = st.sidebar.radio(
    "Mode de pondération",
    ["🤖 Automatique (ML)", "👨‍⚕️ Manuel (Expert)"],
    help="Automatique : calculé par le modèle ML | Manuel : poids définis par expertise épidémiologique"
)

poids_manuels = {}
poids_normalises = {}

if mode_importance == "👨‍⚕️ Manuel (Expert)":
    with st.sidebar.expander("⚙️ Configurer les poids", expanded=True):
        st.markdown("**Définissez l'importance de chaque groupe de variables**")
        st.caption("Les poids seront automatiquement normalisés pour totaliser 100%")
        
        poids_manuels["Historique_Cas"] = st.slider(
            "📈 Historique des cas (lags)",
            min_value=0,
            max_value=100,
            value=40,
            step=5,
            help="Importance des cas passés (4 dernières semaines)"
        )
        
        poids_manuels["Vaccination"] = st.slider(
            "💉 Couverture vaccinale",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            help="Importance du taux de vaccination et non-vaccinés"
        )
        
        poids_manuels["Demographie"] = st.slider(
            "👥 Démographie",
            min_value=0,
            max_value=100,
            value=15,
            step=5,
            help="Importance de la population et densité"
        )
        
        poids_manuels["Urbanisation"] = st.slider(
            "🏙️ Urbanisation",
            min_value=0,
            max_value=100,
            value=8,
            step=2,
            help="Importance du type d'habitat (urbain/rural)"
        )
        
        poids_manuels["Climat"] = st.slider(
            "🌡️ Facteurs climatiques",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
            help="Importance de la température, humidité, saison"
        )
        
        # Calculer le total et normaliser
        total_poids = sum(poids_manuels.values())
        
        if total_poids > 0:
            for key in poids_manuels:
                poids_normalises[key] = poids_manuels[key] / total_poids
        
        # Afficher le résumé
        st.markdown("---")
        st.markdown("**📊 Répartition normalisée :**")
        for key, value in poids_normalises.items():
            st.markdown(f"• {key} : **{value*100:.1f}%**")
        
        if abs(total_poids - 100) > 5:
            st.info(f"ℹ️ Total brut : {total_poids}% → Normalisé à 100%")
else:
    st.sidebar.info("Le modèle ML calculera automatiquement l'importance optimale de chaque variable")

# Seuils d'alerte
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

# Fonctions de chargement géographique
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
            
            iso3_col = None
            for col in ['iso3', 'ISO3', 'iso_code', 'ISO_CODE', 'country_iso', 'COUNTRY_ISO']:
                if col in gdf_full.columns:
                    iso3_col = col
                    break
            
            if iso3_col is None:
                st.warning(f"⚠️ Colonne ISO3 non trouvée. Colonnes : {list(gdf_full.columns)}")
                return gpd.GeoDataFrame()
            
            gdf = gdf_full[gdf_full[iso3_col] == iso3_filter].copy()
            
            if gdf.empty:
                st.warning(f"⚠️ Aucune aire de santé pour {iso3_filter}")
                return gpd.GeoDataFrame()
            
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
        st.error(f"❌ Erreur ZIP : {e}")
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
        st.error(f"❌ Erreur lecture : {e}")
        return gpd.GeoDataFrame()

# ============================================================
# PARTIE 2/6 - CHARGEMENT AIRES DE SANTÉ ET DONNÉES DE CAS
# ============================================================

# Chargement des aires de santé
if st.session_state.sa_gdf_cache is not None and option_aire == "Fichier local (ao_hlthArea.zip)":
    sa_gdf = st.session_state.sa_gdf_cache
    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (cache)")
else:
    with st.spinner(f"🔄 Chargement des aires de santé..."):
        if option_aire == "Fichier local (ao_hlthArea.zip)":
            zip_path = os.path.join("data", "ao_hlthArea.zip")
            if not os.path.exists(zip_path):
                st.error(f"❌ Fichier non trouvé : {zip_path}")
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

# Génération de données fictives
@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500, start=None, end=None):
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
    np.random.seed(42)
    
    return pd.DataFrame({
        "health_area": _sa_gdf["health_area"],
        "Taux_Vaccination": np.random.beta(a=8, b=2, size=len(_sa_gdf)) * 100
    })

# Chargement des données de cas
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
            
            if "Semaine_Epi" in df_raw.columns and "Cas_Total" in df_raw.columns:
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
                df = df_raw.copy()
                
                for col in ["Date_Debut_Eruption", "Date_Notification"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                st.error("❌ Format CSV non reconnu")
                st.stop()
            
            st.sidebar.success(f"✓ {len(df)} cas chargés")
            
        except Exception as e:
            st.error(f"❌ Erreur CSV : {e}")
            st.stop()
        
        if vaccination_file is not None:
            try:
                vaccination_df = pd.read_csv(vaccination_file)
                st.sidebar.success(f"✓ Couverture vaccinale chargée ({len(vaccination_df)} aires)")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erreur vaccination CSV : {e}")
                vaccination_df = None
        else:
            if "Statut_Vaccinal" in df.columns:
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

# Normaliser les noms de colonnes du DataFrame df
COLONNES_MAPPING = {
    # Colonne aire de santé
    "Aire_Sante": ["Aire_Sante", "aire_sante", "health_area", "HEALTH_AREA", "name_fr", "NAME", "nom", "NOM"],
    # Colonne date de début
    "Date_Debut_Eruption": ["Date_Debut_Eruption", "date_debut_eruption", "Date_Debut", "date_onset", "Date_Onset", "symptom_onset"],
    # Colonne date notification
    "Date_Notification": ["Date_Notification", "date_notification", "Date_Notif", "date_notif", "notification_date"],
    # Colonne ID cas
    "ID_Cas": ["ID_Cas", "id_cas", "ID", "id", "Case_ID", "case_id", "ID_cas"],
    # Colonne âge
    "Age_Mois": ["Age_Mois", "age_mois", "Age", "age", "AGE", "Age_Months", "age_months"],
    # Colonne statut vaccinal
    "Statut_Vaccinal": ["Statut_Vaccinal", "statut_vaccinal", "Vaccin", "vaccin", "Vaccination_Status", "vaccination_status", "Vacc_Statut"],
    # Colonne sexe
    "Sexe": ["Sexe", "sexe", "Sex", "sex", "Gender", "gender"],
    # Colonne issue
    "Issue": ["Issue", "issue", "Outcome", "outcome", "OUTCOME"]
}

def normaliser_colonnes(dataframe, mapping):
    """Renommer les colonnes du dataframe selon le mapping standardisé"""
    rename_dict = {}
    for col_standard, col_possibles in mapping.items():
        for col_possible in col_possibles:
            if col_possible in dataframe.columns and col_possible != col_standard:
                rename_dict[col_possible] = col_standard
                break
    if rename_dict:
        dataframe = dataframe.rename(columns=rename_dict)
    return dataframe

# Appliquer la normalisation
df = normaliser_colonnes(df, COLONNES_MAPPING)

# Si "ID_Cas" n'existe pas, en créer une
if "ID_Cas" not in df.columns:
    df["ID_Cas"] = range(1, len(df) + 1)

# Si "Aire_Sante" n'existe pas, essayer de la créer depuis sa_gdf
if "Aire_Sante" not in df.columns:
    # Chercher n'importe quelle colonne qui pourrait contenir un nom d'aire
    for col in df.columns:
        if df[col].dtype == object:
            # Vérifier si les valeurs matchent avec les aires de santé
            sample_values = set(df[col].dropna().unique())
            sa_values = set(sa_gdf["health_area"].unique())
            if len(sample_values.intersection(sa_values)) > 0:
                df["Aire_Sante"] = df[col]
                st.sidebar.info(f"ℹ️ Colonne 'Aire_Sante' créée depuis '{col}'")
                break
    else:
        # Si rien ne match, assigner une aire par défaut
        df["Aire_Sante"] = sa_gdf["health_area"].iloc[0]
        st.sidebar.warning("⚠️ Aucune colonne aire trouvée, valeur par défaut assignée")

# Vérifier et convertir les dates
if "Date_Debut_Eruption" in df.columns:
    df["Date_Debut_Eruption"] = pd.to_datetime(df["Date_Debut_Eruption"], errors='coerce')
else:
    # Chercher une colonne date
    for col in df.columns:
        try:
            test_dates = pd.to_datetime(df[col], errors='coerce')
            if test_dates.notna().sum() > len(df) * 0.5:  # Plus de 50% de dates valides
                df["Date_Debut_Eruption"] = test_dates
                st.sidebar.info(f"ℹ️ 'Date_Debut_Eruption' créée depuis '{col}'")
                break
        except:
            continue
    else:
        # Créer une date par défaut
        df["Date_Debut_Eruption"] = pd.to_datetime(start_date)
        st.sidebar.warning("⚠️ Aucune colonne date trouvée, date de début assignée par défaut")

if "Date_Notification" not in df.columns:
    # Créer Date_Notification = Date_Debut_Eruption + 3 jours par défaut
    df["Date_Notification"] = df["Date_Debut_Eruption"] + pd.to_timedelta(3, unit="D")

# Ajouter des colonnes optionnelles par défaut si absentes
if "Age_Mois" not in df.columns:
    df["Age_Mois"] = np.nan

if "Statut_Vaccinal" not in df.columns:
    df["Statut_Vaccinal"] = "Inconnu"

if "Sexe" not in df.columns:
    df["Sexe"] = "Inconnu"

if "Issue" not in df.columns:
    df["Issue"] = "Inconnu"

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

st.sidebar.info(f"📅 Dernière semaine : **S{derniere_semaine_epi}** ({derniere_annee})")
# ============================================================
# PARTIE 3/6 - ENRICHISSEMENT AVEC DONNÉES EXTERNES
# WorldPop, NASA POWER, GHSL
# ============================================================

# WorldPop - Données démographiques
@st.cache_data
def worldpop_children_stats(_sa_gdf, use_gee):
    if not use_gee:
        st.sidebar.warning("⚠️ WorldPop : GEE indisponible")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Garcons": [np.nan] * len(_sa_gdf),
            "Pop_Filles": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf),
            # Nouvelles colonnes pour pyramide
            "Pop_M_0": [np.nan] * len(_sa_gdf),
            "Pop_M_1": [np.nan] * len(_sa_gdf),
            "Pop_M_5": [np.nan] * len(_sa_gdf),
            "Pop_M_10": [np.nan] * len(_sa_gdf),
            "Pop_F_0": [np.nan] * len(_sa_gdf),
            "Pop_F_1": [np.nan] * len(_sa_gdf),
            "Pop_F_5": [np.nan] * len(_sa_gdf),
            "Pop_F_10": [np.nan] * len(_sa_gdf)
        })
    
    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("📥 Chargement WorldPop...")
        dataset = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
        pop_img = dataset.mosaic()
        
        male_bands = ["M_0", "M_1", "M_5", "M_10"]
        female_bands = ["F_0", "F_1", "F_5", "F_10"]
        
        selected_males = pop_img.select(male_bands)
        selected_females = pop_img.select(female_bands)
        total_pop = pop_img.select(['population'])
        
        # Sommes par sexe
        males_sum = selected_males.reduce(ee.Reducer.sum()).rename('garcons')
        females_sum = selected_females.reduce(ee.Reducer.sum()).rename('filles')
        enfants = males_sum.add(females_sum).rename('enfants')
        
        # ========== MOSAÏQUE AVEC TOUTES LES BANDES ==========
        final_mosaic = (total_pop
                       .addBands(selected_males)      # Bandes M_0, M_1, M_5, M_10
                       .addBands(selected_females)    # Bandes F_0, F_1, F_5, F_10
                       .addBands(males_sum)
                       .addBands(females_sum)
                       .addBands(enfants))
        # ====================================================
        
        # Conversion densité → compte absolu
        pixel_area = ee.Image.pixelArea().divide(10000)
        final_mosaic_count = final_mosaic.multiply(pixel_area)
        
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
        
        status_text.text("🔢 Calcul statistiques zonales...")
        stats = final_mosaic_count.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.sum(),
            scale=100,
            crs='EPSG:4326'
        )
        
        status_text.text("📊 Extraction résultats...")
        stats_info = stats.getInfo()
        
        data_list = []
        total_aires = len(stats_info['features'])
        
        for i, feat in enumerate(stats_info['features']):
            props = feat['properties']
            
            # ========== EXTRACTION DÉTAILLÉE ==========
            pop_totale = props.get("population", 0)
            garcons = props.get("garcons", 0)
            filles = props.get("filles", 0)
            enfants_total = props.get("enfants", 0)
            
            # Extraire chaque tranche d'âge individuellement
            m_0 = props.get("M_0", 0)
            m_1 = props.get("M_1", 0)
            m_5 = props.get("M_5", 0)
            m_10 = props.get("M_10", 0)
            
            f_0 = props.get("F_0", 0)
            f_1 = props.get("F_1", 0)
            f_5 = props.get("F_5", 0)
            f_10 = props.get("F_10", 0)
            # ==========================================
            
            data_list.append({
                "health_area": props.get("health_area", ""),
                "Pop_Totale": int(pop_totale) if pop_totale > 0 else np.nan,
                "Pop_Garcons": int(garcons),
                "Pop_Filles": int(filles),
                "Pop_Enfants": int(enfants_total),
                # Nouvelles colonnes pour pyramide
                "Pop_M_0": int(m_0),
                "Pop_M_1": int(m_1),
                "Pop_M_5": int(m_5),
                "Pop_M_10": int(m_10),
                "Pop_F_0": int(f_0),
                "Pop_F_1": int(f_1),
                "Pop_F_5": int(f_5),
                "Pop_F_10": int(f_10)
            })
            
            progress_value = min((i + 1) / total_aires, 1.0)
            progress_bar.progress(progress_value)
        
        progress_bar.empty()
        status_text.text("✅ WorldPop terminé")
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.error(f"❌ WorldPop : {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Garcons": [np.nan] * len(_sa_gdf),
            "Pop_Filles": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf),
            "Pop_M_0": [np.nan] * len(_sa_gdf),
            "Pop_M_1": [np.nan] * len(_sa_gdf),
            "Pop_M_5": [np.nan] * len(_sa_gdf),
            "Pop_M_10": [np.nan] * len(_sa_gdf),
            "Pop_F_0": [np.nan] * len(_sa_gdf),
            "Pop_F_1": [np.nan] * len(_sa_gdf),
            "Pop_F_5": [np.nan] * len(_sa_gdf),
            "Pop_F_10": [np.nan] * len(_sa_gdf)
        })


# GHSL - Classification urbaine
@st.cache_data
def urban_classification(_sa_gdf, use_gee):
    if not use_gee:
        st.sidebar.warning("⚠️ GHSL : GEE indisponible")
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
            progress_value = min((i + 1) / total_aires, 1.0)
            progress_bar.progress(progress_value)
        
        progress_bar.empty()
        status_text.text("✅ GHSL terminé")
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.error(f"❌ GHSL : {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": [np.nan] * len(_sa_gdf)
        })

# NASA POWER - Données climatiques
@st.cache_data(ttl=86400)
def fetch_climate_nasa_power(_sa_gdf, start_date, end_date):
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
        
        progress_value = min((idx + 1) / total_aires, 1.0)
        progress_bar.progress(progress_value)
    
    progress_bar.empty()
    status_text.text("✅ Climat terminé")
    
    return pd.DataFrame(data_list)

# Enrichissement du GeoDataFrame
with st.spinner("🔄 Enrichissement des données..."):
    pop_df = worldpop_children_stats(sa_gdf, gee_ok)
    urban_df = urban_classification(sa_gdf, gee_ok)
    climate_df = fetch_climate_nasa_power(sa_gdf, start_date, end_date)

sa_gdf_enrichi = sa_gdf.copy()
sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df, on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df, on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df, on="health_area", how="left")

if vaccination_df is not None:
    sa_gdf_enrichi = sa_gdf_enrichi.merge(vaccination_df, on="health_area", how="left")
else:
    sa_gdf_enrichi["Taux_Vaccination"] = np.nan

# reprojection en métrique égale-aire
sa_gdf_m = sa_gdf_enrichi.to_crs("ESRI:54009")  # Mollweide (mètres)

sa_gdf_enrichi["Superficie_km2"] = sa_gdf_m.geometry.area / 1e6

sa_gdf_enrichi["Densite_Pop"] = (
    sa_gdf_enrichi["Pop_Totale"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

sa_gdf_enrichi["Densite_Enfants"] = (
    sa_gdf_enrichi["Pop_Enfants"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

sa_gdf_enrichi = sa_gdf_enrichi.replace([np.inf, -np.inf], np.nan)

st.sidebar.success("✓ Enrichissement terminé")

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
# PARTIE 4/6 - KPIS, CARTE ET ANALYSES
# ============================================================

# KPIs
st.header("📊 Indicateurs Clés de Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📈 Cas totaux", f"{len(df):,}")

with col2:
    if "Statut_Vaccinal" in df.columns and df["Statut_Vaccinal"].notna().sum() > 0 and (df["Statut_Vaccinal"] != "Inconnu").sum() > 0:
        taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
        delta_vac = taux_non_vac - 45
        st.metric("💉 Non vaccinés", f"{taux_non_vac:.1f}%", delta=f"{delta_vac:+.1f}%")
    else:
        st.metric("💉 Non vaccinés", "N/A")

with col3:
    if "Age_Mois" in df.columns and df["Age_Mois"].notna().sum() > 0:
        age_median = df["Age_Mois"].median()
        st.metric("👶 Âge médian", f"{int(age_median)} mois")
    else:
        st.metric("👶 Âge médian", "N/A")

with col4:
    if "Issue" in df.columns and df["Issue"].notna().sum() > 0 and (df["Issue"] == "Décédé").sum() > 0:
        taux_deces = (df["Issue"] == "Décédé").mean() * 100
        st.metric("☠️ Létalité", f"{taux_deces:.2f}%")
    else:
        st.metric("☠️ Létalité", "N/A")

with col5:
    n_aires_touchees = df["Aire_Sante"].nunique()
    pct_aires = (n_aires_touchees / len(sa_gdf)) * 100
    st.metric("🗺️ Aires touchées", f"{n_aires_touchees}/{len(sa_gdf)}", delta=f"{pct_aires:.0f}%")

# Agrégation par aire
agg_dict = {"ID_Cas": "count"}

if "Age_Mois" in df.columns:
    agg_dict["Age_Mois"] = "mean"

if "Statut_Vaccinal" in df.columns:
    agg_dict["Statut_Vaccinal"] = lambda x: (x == "Non").mean() * 100

cases_by_area = df.groupby("Aire_Sante").agg(agg_dict).reset_index()

# Renommer les colonnes selon ce qui est présent
rename_map = {"ID_Cas": "Cas_Observes"}
if "Age_Mois" in cases_by_area.columns:
    rename_map["Age_Mois"] = "Age_Moyen"
if "Statut_Vaccinal" in cases_by_area.columns:
    rename_map["Statut_Vaccinal"] = "Taux_Non_Vaccines"

cases_by_area = cases_by_area.rename(columns=rename_map)

# Ajouter des colonnes par défaut si absentes
if "Taux_Non_Vaccines" not in cases_by_area.columns:
    cases_by_area["Taux_Non_Vaccines"] = 0
if "Age_Moyen" not in cases_by_area.columns:
    cases_by_area["Age_Moyen"] = 0

cases_by_area.columns = ["Aire_Sante", "Cas_Observes", "Taux_Non_Vaccines", "Age_Moyen"]

sa_gdf_with_cases = sa_gdf_enrichi.merge(
    cases_by_area,
    left_on="health_area",
    right_on="Aire_Sante",
    how="left"
)

sa_gdf_with_cases["Cas_Observes"] = sa_gdf_with_cases["Cas_Observes"].fillna(0)
sa_gdf_with_cases["Taux_Non_Vaccines"] = sa_gdf_with_cases["Taux_Non_Vaccines"].fillna(0)

sa_gdf_with_cases["Taux_Attaque_10000"] = (
    sa_gdf_with_cases["Cas_Observes"] /
    sa_gdf_with_cases["Pop_Enfants"].replace(0, np.nan) * 10000
).replace([np.inf, -np.inf], np.nan)

# Carte de situation actuelle
st.header("🗺️ Cartographie de la Situation Actuelle")

center_lat = sa_gdf_with_cases.geometry.centroid.y.mean()
center_lon = sa_gdf_with_cases.geometry.centroid.x.mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB positron",
    control_scale=True
)

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

for idx, row in sa_gdf_with_cases.iterrows():
    aire_name = row['health_area']
    cas_obs = int(row.get('Cas_Observes', 0))
    pop_enfants = row.get('Pop_Enfants', np.nan)
    taux_attaque = row.get('Taux_Attaque_10000', np.nan)
    urbanisation = row.get('Urbanisation', 'N/A')
    densite = row.get('Densite_Pop', np.nan)
    
    popup_html = f"""
    <div style="font-family: Arial; width: 350px;">
        <h3 style="margin-bottom: 10px; color: #1976d2; border-bottom: 2px solid #1976d2;">
            {aire_name}
        </h3>
        <div style="background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin: 0; color: #d32f2f;">📊 Situation Épidémiologique</h4>
            <table style="width: 100%; margin-top: 5px;">
                <tr><td><b>Cas observés :</b></td><td style="text-align: right;">
                    <b style="font-size: 18px; color: #d32f2f;">{cas_obs}</b>
                </td></tr>
                <tr><td>Population enfants :</td><td style="text-align: right;">
                    {f"{int(pop_enfants):,}" if not np.isnan(pop_enfants) else "N/A"}
                </td></tr>
                <tr><td>Taux d'attaque :</td><td style="text-align: right;">
                    {f"{taux_attaque:.1f}/10K" if not np.isnan(taux_attaque) else "N/A"}
                </td></tr>
                <tr><td>Type habitat :</td><td style="text-align: right;">
                    <b>{urbanisation if pd.notna(urbanisation) else "N/A"}</b>
                </td></tr>
                <tr><td>Densité pop :</td><td style="text-align: right;">
                    {f"{densite:.1f} hab/km²" if not np.isnan(densite) else "N/A"}
                </td></tr>
            </table>
        </div>
    </div>
    """
    
    fill_color = colormap(row['Cas_Observes']) if max_cases > 0 else '#e0e0e0'
    
    if row['Cas_Observes'] >= seuil_alerte_epidemique:
        line_color = '#b71c1c'
        line_weight = 2
    else:
        line_color = 'black'
        line_weight = 0.5
    
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
    
    if cas_obs > 0:
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 9pt;
                    color: black;
                    font-weight: normal;
                    background: none;
                    padding: 0;
                    border: none;
                    box-shadow: none;
                    white-space: nowrap;
                ">{aire_name}</div>
                """
            ),
        ).add_to(m)


heat_data = [
    [row.geometry.centroid.y, row.geometry.centroid.x, row['Cas_Observes']]
    for idx, row in sa_gdf_with_cases.iterrows()
    if row['Cas_Observes'] > 0
]

if heat_data:
    HeatMap(
        heat_data,
        radius=20,
        blur=25,
        max_zoom=13,
        gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'}
    ).add_to(m)

legend_html = f'''
<div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 250px;
    background-color: white;
    border: 2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    border-radius: 5px;">
    <p style="margin: 0; font-weight: bold;">📊 Légende</p>
    <p style="margin: 5px 0;">
        <span style="background-color: #e8f5e9; padding: 2px 8px;">Faible</span>
        0-{max_cases//3:.0f} cas
    </p>
    <p style="margin: 5px 0;">
        <span style="background-color: #ffeb3b; padding: 2px 8px;">Moyen</span>
        {max_cases//3:.0f}-{2*max_cases//3:.0f} cas
    </p>
    <p style="margin: 5px 0;">
        <span style="background-color: #f44336; padding: 2px 8px; color: white;">Élevé</span>
        >{2*max_cases//3:.0f} cas
    </p>
    <p style="margin: 5px 0; padding-top: 5px; border-top: 1px solid #ccc;">
        <b>Seuil alerte :</b> {seuil_alerte_epidemique} cas/sem
    </p>
</div>
'''

m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=1400, height=650)

col1, col2, col3 = st.columns(3)

with col1:
    aires_alerte = len(sa_gdf_with_cases[sa_gdf_with_cases['Cas_Observes'] >= seuil_alerte_epidemique])
    st.metric("🚨 Aires en alerte", aires_alerte, f"{aires_alerte/len(sa_gdf)*100:.1f}%")

with col2:
    aires_sans_cas = len(sa_gdf_with_cases[sa_gdf_with_cases['Cas_Observes'] == 0])
    st.metric("✅ Aires sans cas", aires_sans_cas, f"{aires_sans_cas/len(sa_gdf)*100:.1f}%")

with col3:
    densite_pop_moy = sa_gdf_with_cases['Densite_Pop'].mean()
    st.metric("📍 Densité pop. moy.", f"{densite_pop_moy:.1f} hab/km²")
# Analyse temporelle
st.header("📈 Analyse Temporelle par Semaines Épidémiologiques")

weekly_cases = df.groupby(['Annee', 'Semaine_Epi']).size().reset_index(name='Cas')
weekly_cases['Semaine_Label'] = weekly_cases['Annee'].astype(str) + '-S' + weekly_cases['Semaine_Epi'].astype(str).str.zfill(2)

fig_epi = go.Figure()

fig_epi.add_trace(go.Scatter(
    x=weekly_cases['Semaine_Label'],
    y=weekly_cases['Cas'],
    mode='lines+markers',
    name='Cas observés',
    line=dict(color='#d32f2f', width=3),
    marker=dict(size=6),
    hovertemplate='<b>%{x}</b><br>Cas : %{y}<extra></extra>'
))

from scipy.signal import savgol_filter

if len(weekly_cases) > 5:
    tendance = savgol_filter(
        weekly_cases['Cas'],
        window_length=min(7, len(weekly_cases) if len(weekly_cases) % 2 == 1 else len(weekly_cases)-1),
        polyorder=2
    )
    fig_epi.add_trace(go.Scatter(
        x=weekly_cases['Semaine_Label'],
        y=tendance,
        mode='lines',
        name='Tendance',
        line=dict(color='#1976d2', width=2, dash='dash'),
        hovertemplate='<b>%{x}</b><br>Tendance : %{y:.1f}<extra></extra>'
    ))

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
    st.metric("📊 Moyenne hebdomadaire", f"{cas_moyen_semaine:.1f} cas")

with col3:
    if len(weekly_cases) >= 2:
        variation = weekly_cases.iloc[-1]['Cas'] - weekly_cases.iloc[-2]['Cas']
        cas_precedent = weekly_cases.iloc[-2]['Cas']
        pct_variation = (variation / cas_precedent * 100) if cas_precedent > 0 else 0
        st.metric("📉 Variation dernière semaine", f"{int(variation):+d} cas", f"{pct_variation:+.1f}%")
    else:
        st.metric("📉 Variation dernière semaine", "N/A")

# Distribution par âge
st.subheader("👶 Distribution par Tranches d'Âge")

if "Age_Mois" in df.columns:
    df["Tranche_Age"] = pd.cut(
        df["Age_Mois"],
        bins=[0, 12, 60, 120, 180],
        labels=["0-1 an", "1-5 ans", "5-10 ans", "10-15 ans"]
    )
    age_available = True
else:
    df["Tranche_Age"] = "Inconnu"
    age_available = False

agg_dict_age = {"ID_Cas": "count"}

if "Statut_Vaccinal" in df.columns:
    agg_dict_age["Statut_Vaccinal"] = lambda x: (x == "Non").mean() * 100

age_stats = df.groupby("Tranche_Age").agg(agg_dict_age).reset_index()

rename_map_age = {"ID_Cas": "Nombre_Cas"}
if "Statut_Vaccinal" in age_stats.columns:
    rename_map_age["Statut_Vaccinal"] = "Pct_Non_Vaccines"

age_stats = age_stats.rename(columns=rename_map_age)

if "Pct_Non_Vaccines" not in age_stats.columns:
    age_stats["Pct_Non_Vaccines"] = 0

col1, col2 = st.columns(2)

col1, col2 = st.columns(2)

with col1:
    if age_available:
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
    else:
        st.info("ℹ️ Données d'âge non disponibles")

with col2:
    if age_available and age_stats["Pct_Non_Vaccines"].sum() > 0:
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
    else:
        st.info("ℹ️ Données de vaccination par âge non disponibles")
# ============================================================
# PYRAMIDE DES ÂGES POPULATION (WORLDPOP)
# ============================================================

st.header("📊 Pyramide des Âges - Population Enfantine")

if donnees_dispo["Population"]:
    # Préparer les données de population par tranche d'âge
    pyramid_data = []
    
    for idx, row in sa_gdf_enrichi.iterrows():
        aire = row['health_area']
        
        # Récupérer les valeurs de population par tranche d'âge depuis WorldPop
        # Si vous avez déjà extrait ces données individuellement
        pop_0_1_m = row.get('Pop_M_0', 0) + row.get('Pop_M_1', 0)  # 0-4 ans garçons
        pop_5_9_m = row.get('Pop_M_5', 0)  # 5-9 ans garçons
        pop_10_14_m = row.get('Pop_M_10', 0)  # 10-14 ans garçons
        
        pop_0_1_f = row.get('Pop_F_0', 0) + row.get('Pop_F_1', 0)  # 0-4 ans filles
        pop_5_9_f = row.get('Pop_F_5', 0)  # 5-9 ans filles
        pop_10_14_f = row.get('Pop_F_10', 0)  # 10-14 ans filles
        
        pyramid_data.append({
            'Aire': aire,
            'Garçons_0-4': pop_0_1_m,
            'Garçons_5-9': pop_5_9_m,
            'Garçons_10-14': pop_10_14_m,
            'Filles_0-4': pop_0_1_f,
            'Filles_5-9': pop_5_9_f,
            'Filles_10-14': pop_10_14_f
        })
    
    pyramid_df = pd.DataFrame(pyramid_data)
    
    # Agréger pour tout le territoire
    total_garcons_0_4 = pyramid_df['Garçons_0-4'].sum()
    total_garcons_5_9 = pyramid_df['Garçons_5-9'].sum()
    total_garcons_10_14 = pyramid_df['Garçons_10-14'].sum()
    
    total_filles_0_4 = pyramid_df['Filles_0-4'].sum()
    total_filles_5_9 = pyramid_df['Filles_5-9'].sum()
    total_filles_10_14 = pyramid_df['Filles_10-14'].sum()
    
    # Créer le dataframe pour la pyramide
    age_groups = ['0-4', '5-9', '10-14']
    
    pyramid_plot_df = pd.DataFrame({
        'Age': age_groups,
        'Garçons': [-total_garcons_0_4, -total_garcons_5_9, -total_garcons_10_14],  # Négatif pour gauche
        'Filles': [total_filles_0_4, total_filles_5_9, total_filles_10_14]  # Positif pour droite
    })
    
    # Créer la pyramide avec Plotly
    fig_pyramid = go.Figure()
    
    # Barres garçons (gauche - négatif)
    fig_pyramid.add_trace(go.Bar(
        y=pyramid_plot_df['Age'],
        x=pyramid_plot_df['Garçons'],
        name='Garçons',
        orientation='h',
        marker=dict(color='#42a5f5'),
        text=[f"{abs(x):,.0f}" for x in pyramid_plot_df['Garçons']],
        textposition='inside',
        hovertemplate='<b>%{y} ans</b><br>Garçons: %{text}<extra></extra>'
    ))
    
    # Barres filles (droite - positif)
    fig_pyramid.add_trace(go.Bar(
        y=pyramid_plot_df['Age'],
        x=pyramid_plot_df['Filles'],
        name='Filles',
        orientation='h',
        marker=dict(color='#ec407a'),
        text=[f"{x:,.0f}" for x in pyramid_plot_df['Filles']],
        textposition='inside',
        hovertemplate='<b>%{y} ans</b><br>Filles: %{text}<extra></extra>'
    ))
    
    # Calculer les limites symétriques
    max_val = max(
        abs(pyramid_plot_df['Garçons'].min()),
        pyramid_plot_df['Filles'].max()
    )
    
    fig_pyramid.update_layout(
        title='Pyramide des Âges - Population Enfantine (0-14 ans)',
        xaxis=dict(
            title='Population',
            tickvals=[-max_val, -max_val/2, 0, max_val/2, max_val],
            ticktext=[f"{int(max_val):,}", f"{int(max_val/2):,}", "0", 
                     f"{int(max_val/2):,}", f"{int(max_val):,}"],
            range=[-max_val * 1.1, max_val * 1.1]
        ),
        yaxis=dict(title='Tranche d\'âge'),
        barmode='overlay',
        height=400,
        bargap=0.1,
        showlegend=True,
        legend=dict(x=0.85, y=0.95),
        hovermode='y unified'
    )
    
    st.plotly_chart(fig_pyramid, use_container_width=True)
    
    # Statistiques complémentaires
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_garcons = total_garcons_0_4 + total_garcons_5_9 + total_garcons_10_14
        st.metric("👦 Garçons (0-14 ans)", f"{int(total_garcons):,}")
    
    with col2:
        total_filles = total_filles_0_4 + total_filles_5_9 + total_filles_10_14
        st.metric("👧 Filles (0-14 ans)", f"{int(total_filles):,}")
    
    with col3:
        ratio = (total_garcons / total_filles * 100) if total_filles > 0 else 0
        st.metric("⚖️ Ratio G/F", f"{ratio:.1f}%")

else:
    st.info("📊 Données de population non disponibles. Pyramide des âges non affichable.")

# Nowcasting
st.subheader("⏱️ Nowcasting - Correction des Délais de Notification")

st.info("""
**Nowcasting (Prévision immédiate) :** Technique d'ajustement permettant d'estimer le nombre réel de cas en tenant compte des délais de notification.
""")

if "Date_Notification" in df.columns and "Date_Debut_Eruption" in df.columns:
    df["Delai_Notification"] = (df["Date_Notification"] - df["Date_Debut_Eruption"]).dt.days
    delai_available = True
else:
    df["Delai_Notification"] = 3  # Valeur par défaut
    delai_available = False

delai_moyen = df["Delai_Notification"].mean()
delai_median = df["Delai_Notification"].median()
delai_std = df["Delai_Notification"].std()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Délai moyen de notification", f"{delai_moyen:.1f} jours" if delai_available and not np.isnan(delai_moyen) else "N/A")

with col2:
    st.metric("Délai médian", f"{delai_median:.0f} jours" if delai_available and not np.isnan(delai_median) else "N/A")

with col3:
    st.metric("Écart-type", f"{delai_std:.1f} jours" if delai_available and not np.isnan(delai_std) else "N/A")

with col4:
    derniere_semaine_label = weekly_cases.iloc[-1]['Semaine_Label']
    cas_derniere_semaine = int(weekly_cases.iloc[-1]['Cas'])
    if delai_available and not np.isnan(delai_moyen):
        facteur_correction = 1 + (delai_moyen / 7)
        cas_corriges = int(cas_derniere_semaine * facteur_correction)
        st.metric(
            f"Cas corrigés ({derniere_semaine_label})",
            cas_corriges,
            delta=f"+{cas_corriges - cas_derniere_semaine}"
        )
    else:
        st.metric(
            f"Cas corrigés ({derniere_semaine_label})",
            cas_derniere_semaine,
            delta="N/A"
        )

if delai_available:
    fig_delai = px.histogram(
        df,
        x="Delai_Notification",
        nbins=20,
        title="Distribution des délais de notification",
        labels={"Delai_Notification": "Délai (jours)", "count": "Nombre de cas"},
        color_discrete_sequence=['#d32f2f']
    )

    fig_delai.add_vline(x=delai_moyen, line_dash="dash", line_color="blue", annotation_text=f"Moyenne : {delai_moyen:.1f}j")
    fig_delai.add_vline(x=delai_median, line_dash="dash", line_color="green", annotation_text=f"Médiane : {delai_median:.0f}j")

    st.plotly_chart(fig_delai, use_container_width=True)
else:
    st.info("ℹ️ Données de délai de notification non disponibles")

# ============================================================
# PARTIE 5/6 - MODÉLISATION PRÉDICTIVE AVEC SYSTÈME HYBRIDE
# ============================================================

st.header("🔮 Modélisation Prédictive par Semaines Épidémiologiques")

st.markdown(f"""
<div class="info-box">
<b>Configuration de la prédiction :</b><br>
- Dernière semaine de données : <b>S{derniere_semaine_epi} ({derniere_annee})</b><br>
- Période de prédiction : <b>{pred_mois} mois ({n_weeks_pred} semaines)</b><br>
- Semaines prédites : <b>S{derniere_semaine_epi+1} à S{min(derniere_semaine_epi+n_weeks_pred, 52)}</b><br>
- Modèle sélectionné : <b>{modele_choisi}</b><br>
- Mode importance : <b>{mode_importance}</b><br>
- Seuils configurés : Baisse ≥{seuil_baisse}%, Hausse ≥{seuil_hausse}%
</div>
""", unsafe_allow_html=True)

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
    
    with st.spinner("🤖 Préparation des données et entraînement..."):
        
        weekly_features = df.groupby(["Aire_Sante", "Annee", "Semaine_Epi"]).agg(
            Cas_Observes=("ID_Cas", "count"),
            Non_Vaccines=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100),
            Age_Moyen=("Age_Mois", "mean")
        ).reset_index()
        
        weekly_features['Semaine_Label'] = (
            weekly_features['Annee'].astype(str) + '-S' +
            weekly_features['Semaine_Epi'].astype(str).str.zfill(2)
        )
        
        weekly_features = weekly_features.merge(
            sa_gdf_enrichi[[
                "health_area", "Pop_Totale", "Pop_Enfants",
                "Densite_Pop", "Densite_Enfants", "Urbanisation",
                "Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite",
                "Taux_Vaccination"
            ]],
            left_on="Aire_Sante",
            right_on="health_area",
            how="left"
        )
        
        le_urban = LabelEncoder()
        weekly_features["Urban_Encoded"] = le_urban.fit_transform(
            weekly_features["Urbanisation"].fillna("Rural")
        )
        
        # Créer un coefficient climatique composite si données disponibles
        if donnees_dispo["Climat"]:
            scaler_climat = MinMaxScaler()
            climate_cols = ["Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite"]
            
            for col in climate_cols:
                if col in weekly_features.columns:
                    weekly_features[f"{col}_Norm"] = scaler_climat.fit_transform(
                        weekly_features[[col]].fillna(weekly_features[col].mean())
                    )
            
            weekly_features["Coef_Climatique"] = (
                weekly_features.get("Temperature_Moy_Norm", 0) * 0.4 +
                weekly_features.get("Humidite_Moy_Norm", 0) * 0.4 +
                weekly_features.get("Saison_Seche_Humidite_Norm", 0) * 0.2
            )
        
        weekly_features = weekly_features.sort_values(['Aire_Sante', 'Annee', 'Semaine_Epi'])
        
        for lag in [1, 2, 3, 4]:
            weekly_features[f'Cas_Lag_{lag}'] = (
                weekly_features.groupby('Aire_Sante')['Cas_Observes'].shift(lag)
            )
        
        numeric_cols = weekly_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            weekly_features[col] = weekly_features[col].replace([np.inf, -np.inf], 0)
            mean_val = weekly_features[col].mean()
            weekly_features[col] = weekly_features[col].fillna(mean_val if pd.notna(mean_val) else 0)
        
        st.subheader("📚 Entraînement du Modèle")
        
        # Construire la liste des features en fonction des données disponibles
        feature_cols = [
            "Cas_Observes", "Age_Moyen", "Semaine_Epi",
            "Cas_Lag_1", "Cas_Lag_2", "Cas_Lag_3", "Cas_Lag_4"
        ]
        
        # Mapping des groupes vers les colonnes individuelles
        feature_groups = {
            "Historique_Cas": ["Cas_Lag_1", "Cas_Lag_2", "Cas_Lag_3", "Cas_Lag_4"],
            "Vaccination": [],
            "Demographie": [],
            "Urbanisation": [],
            "Climat": []
        }
        
        if donnees_dispo["Population"]:
            feature_cols.extend(["Pop_Totale", "Pop_Enfants", "Densite_Pop", "Densite_Enfants"])
            feature_groups["Demographie"] = ["Pop_Totale", "Pop_Enfants", "Densite_Pop", "Densite_Enfants"]
            st.info("✅ Données démographiques intégrées au modèle")
        
        if donnees_dispo["Urbanisation"]:
            feature_cols.append("Urban_Encoded")
            feature_groups["Urbanisation"] = ["Urban_Encoded"]
            st.info("✅ Classification urbaine intégrée au modèle")
        
        if donnees_dispo["Climat"]:
            feature_cols.append("Coef_Climatique")
            feature_groups["Climat"] = ["Coef_Climatique"]
            st.info("✅ Coefficient climatique composite intégré au modèle")
        
        if donnees_dispo["Vaccination"]:
            feature_cols.extend(["Taux_Vaccination", "Non_Vaccines"])
            feature_groups["Vaccination"] = ["Taux_Vaccination", "Non_Vaccines"]
            st.info("✅ Données vaccinales intégrées au modèle")
        elif "Non_Vaccines" in weekly_features.columns:
            feature_cols.append("Non_Vaccines")
            feature_groups["Vaccination"] = ["Non_Vaccines"]
        
        st.markdown(f"**Variables utilisées :** {len(feature_cols)} features")
        
        weekly_features_clean = weekly_features.dropna(subset=feature_cols)
        
        if len(weekly_features_clean) < 20:
            st.warning("⚠️ Données insuffisantes (minimum 20 observations requises)")
            st.stop()
        
        X = weekly_features_clean[feature_cols].copy()
        y = weekly_features_clean["Cas_Observes"]
        
        # ========== APPLICATION DES POIDS (MODE MANUEL) ==========
        if mode_importance == "👨‍⚕️ Manuel (Expert)":
            st.markdown('<div class="weight-box">', unsafe_allow_html=True)
            st.markdown("**⚖️ Application des poids manuels aux variables**")
            
            # Créer un dictionnaire de mapping colonne → poids
            column_weights = {}
            
            for group_name, weight in poids_normalises.items():
                if group_name in feature_groups:
                    cols_in_group = feature_groups[group_name]
                    if len(cols_in_group) > 0:
                        weight_per_col = weight / len(cols_in_group)
                        for col in cols_in_group:
                            if col in feature_cols:
                                column_weights[col] = weight_per_col
            
            # Ajouter des poids par défaut pour les colonnes non groupées
            for col in feature_cols:
                if col not in column_weights:
                    column_weights[col] = 0.01
            
            # Appliquer les poids aux features
            X_weighted = X.copy()
            for col in feature_cols:
                if col in column_weights:
                    X_weighted[col] = X_weighted[col] * column_weights[col]
            
            # Afficher le détail des poids appliqués
            weights_df = pd.DataFrame({
                "Variable": list(column_weights.keys()),
                "Poids": [f"{v*100:.2f}%" for v in column_weights.values()]
            })
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
            with col2:
                st.metric("Total des poids", "100.00%")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Utiliser X_weighted pour l'entraînement
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_weighted)
        else:
            # Mode automatique : pas de pondération manuelle
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        if modele_choisi == "GradientBoosting (Recommandé)":
            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=4,
                random_state=42
            )
        elif modele_choisi == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                random_state=42
            )
        elif modele_choisi == "Ridge Regression":
            model = Ridge(alpha=1.0, random_state=42)
        elif modele_choisi == "Lasso Regression":
            model = Lasso(alpha=0.1, random_state=42)
        elif modele_choisi == "Decision Tree":
            model = DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=4,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        
        score_test = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 R² Test", f"{score_test:.3f}")
        with col2:
            st.metric("🎯 R² CV (5-fold)", f"{cv_mean:.3f}")
        with col3:
            st.metric("📏 Écart-type CV", f"±{cv_std:.3f}")
        
        if cv_mean > 0.7:
            st.success(f"✅ Modèle performant ({modele_choisi})")
        elif cv_mean > 0.5:
            st.warning(f"⚠️ Modèle acceptable ({modele_choisi})")
        else:
            st.error(f"❌ Modèle peu performant - envisagez un autre algorithme")
        
        # Afficher l'importance des variables (automatique OU ajustée)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                "Variable": feature_cols,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)
            
            with st.expander("📊 Importance des variables", expanded=True):
                if mode_importance == "👨‍⚕️ Manuel (Expert)":
                    st.info("ℹ️ Ces importances reflètent l'influence des variables **après application des poids manuels**")
                else:
                    st.info("ℹ️ Ces importances sont **calculées automatiquement** par le modèle ML")
                
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
        
        st.subheader(f"📅 Génération des Prédictions - {n_weeks_pred} Semaines")
        
        future_climate = None
        if donnees_dispo["Climat"]:
            future_start = end_date + timedelta(days=1)
            future_end = end_date + timedelta(days=n_weeks_pred * 7)
            
            with st.spinner("🌡️ Chargement prévisions climatiques..."):
                try:
                    future_climate = fetch_climate_nasa_power(sa_gdf, future_start, future_end)
                    st.info("✅ Prévisions climatiques intégrées aux prédictions")
                except:
                    st.warning("⚠️ Prévisions climatiques indisponibles - utilisation valeurs moyennes")
        
        future_predictions = []
        
        for aire in weekly_features["Aire_Sante"].unique():
            aire_data = weekly_features[weekly_features["Aire_Sante"] == aire].sort_values(['Annee', 'Semaine_Epi'])
            
            if aire_data.empty:
                continue
            
            last_obs = aire_data.iloc[-1]
            
            last_4_weeks = aire_data.tail(4)['Cas_Observes'].values
            if len(last_4_weeks) < 4:
                last_4_weeks = np.pad(last_4_weeks, (4-len(last_4_weeks), 0), 'edge')
            
            for i in range(1, n_weeks_pred + 1):
                nouvelle_semaine_epi = (derniere_semaine_epi + i - 1) % 52 + 1
                nouvelle_annee = derniere_annee + ((derniere_semaine_epi + i - 1) // 52)
                
                future_row = {
                    "Aire_Sante": aire,
                    "Annee": nouvelle_annee,
                    "Semaine_Epi": nouvelle_semaine_epi,
                    "Semaine_Label": f"{nouvelle_annee}-S{str(nouvelle_semaine_epi).zfill(2)}",
                    "Age_Moyen": last_obs["Age_Moyen"]
                }
                
                if donnees_dispo["Population"]:
                    future_row.update({
                        "Pop_Totale": last_obs["Pop_Totale"],
                        "Pop_Enfants": last_obs["Pop_Enfants"],
                        "Densite_Pop": last_obs["Densite_Pop"],
                        "Densite_Enfants": last_obs["Densite_Enfants"]
                    })
                
                if donnees_dispo["Urbanisation"]:
                    future_row["Urban_Encoded"] = last_obs["Urban_Encoded"]
                
                if donnees_dispo["Climat"]:
                    if future_climate is not None:
                        climate_aire = future_climate[future_climate["health_area"] == aire]
                        if not climate_aire.empty:
                            temp_norm = scaler_climat.transform([[climate_aire.iloc[0]["Temperature_Moy"]]])[0][0]
                            hum_norm = scaler_climat.transform([[climate_aire.iloc[0]["Humidite_Moy"]]])[0][0]
                            saison_norm = scaler_climat.transform([[climate_aire.iloc[0]["Saison_Seche_Humidite"]]])[0][0]
                            
                            future_row["Coef_Climatique"] = temp_norm * 0.4 + hum_norm * 0.4 + saison_norm * 0.2
                        else:
                            future_row["Coef_Climatique"] = last_obs.get("Coef_Climatique", 0)
                    else:
                        future_row["Coef_Climatique"] = last_obs.get("Coef_Climatique", 0)
                
                if donnees_dispo["Vaccination"]:
                    future_row["Taux_Vaccination"] = last_obs["Taux_Vaccination"]
                    future_row["Non_Vaccines"] = last_obs["Non_Vaccines"]
                elif "Non_Vaccines" in last_obs:
                    future_row["Non_Vaccines"] = last_obs["Non_Vaccines"]
                
                if i == 1:
                    future_row["Cas_Observes"] = last_obs["Cas_Observes"]
                    future_row["Cas_Lag_1"] = last_4_weeks[-1]
                    future_row["Cas_Lag_2"] = last_4_weeks[-2] if len(last_4_weeks) >= 2 else last_4_weeks[-1]
                    future_row["Cas_Lag_3"] = last_4_weeks[-3] if len(last_4_weeks) >= 3 else last_4_weeks[-1]
                    future_row["Cas_Lag_4"] = last_4_weeks[-4] if len(last_4_weeks) >= 4 else last_4_weeks[-1]
                else:
                    prev_predictions = [
                        p["Predicted_Cases"] for p in future_predictions
                        if p["Aire_Sante"] == aire
                    ]
                    future_row["Cas_Observes"] = prev_predictions[-1] if prev_predictions else last_obs["Cas_Observes"]
                    future_row["Cas_Lag_1"] = prev_predictions[-1] if len(prev_predictions) >= 1 else last_4_weeks[-1]
                    future_row["Cas_Lag_2"] = prev_predictions[-2] if len(prev_predictions) >= 2 else last_4_weeks[-2]
                    future_row["Cas_Lag_3"] = prev_predictions[-3] if len(prev_predictions) >= 3 else last_4_weeks[-3]
                    future_row["Cas_Lag_4"] = prev_predictions[-4] if len(prev_predictions) >= 4 else last_4_weeks[-4]
                
                X_future = np.array([[future_row[col] for col in feature_cols]])
                
                # Appliquer les mêmes poids si mode manuel
                if mode_importance == "👨‍⚕️ Manuel (Expert)":
                    for idx, col in enumerate(feature_cols):
                        if col in column_weights:
                            X_future[0, idx] = X_future[0, idx] * column_weights[col]
                
                X_future_scaled = scaler.transform(X_future)
                
                predicted_cases = max(0, model.predict(X_future_scaled)[0])
                
                if cv_std > 0:
                    noise = np.random.normal(0, predicted_cases * cv_std * 0.1)
                    predicted_cases = max(0, predicted_cases + noise)
                
                future_row["Predicted_Cases"] = predicted_cases
                future_predictions.append(future_row)
        
        future_df = pd.DataFrame(future_predictions)
        
        st.success(f"✓ {len(future_df)} prédictions générées ({len(future_df['Aire_Sante'].unique())} aires × {n_weeks_pred} semaines)")
        
        moyenne_historique = weekly_features.groupby("Aire_Sante")["Cas_Observes"].mean().reset_index()
        moyenne_historique.columns = ["Aire_Sante", "Moyenne_Historique"]
        
        risk_df = future_df.groupby("Aire_Sante").agg(
            Cas_Predits_Total=("Predicted_Cases", "sum"),
            Cas_Predits_Max=("Predicted_Cases", "max"),
            Cas_Predits_Moyen=("Predicted_Cases", "mean"),
            Semaine_Pic=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(), "Semaine_Label"] if len(x) > 0 else "N/A")
        ).reset_index()
        
        risk_df = risk_df.merge(moyenne_historique, on="Aire_Sante", how="left")
        
        risk_df["Variation_Pct"] = (
            (risk_df["Cas_Predits_Moyen"] - risk_df["Moyenne_Historique"]) /
            risk_df["Moyenne_Historique"].replace(0, 1)
        ) * 100
        
        risk_df["Categorie_Variation"] = pd.cut(
            risk_df["Variation_Pct"],
            bins=[-np.inf, -seuil_baisse, -10, 10, seuil_hausse, np.inf],
            labels=["Forte baisse", "Baisse modérée", "Stable", "Hausse modérée", "Forte hausse"]
        )
        
        risk_df = risk_df.sort_values("Variation_Pct", ascending=False)

# ============================================================
# PARTIE 6/6 - VISUALISATIONS FINALES ET EXPORTS
# ============================================================

        st.subheader("📊 Synthèse des Prédictions")
        
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
        
        st.subheader("🔥 Heatmap Temporelle des Prédictions")
        
        top_10_hausse = risk_df.head(10)["Aire_Sante"].tolist()
        top_10_baisse = risk_df.tail(10)["Aire_Sante"].tolist()
        
        heatmap_data = future_df.pivot(
            index="Aire_Sante",
            columns="Semaine_Label",
            values="Predicted_Cases"
        ).fillna(0)
        
        col_heat1, col_heat2 = st.columns(2)
        
        with col_heat1:
            st.markdown("**🔴 Top 10 - Hausses prédites**")
            heatmap_hausse = heatmap_data.loc[heatmap_data.index.isin(top_10_hausse)]
            
            fig_heat_hausse = px.imshow(
                heatmap_hausse,
                labels=dict(x="Semaine", y="Aire", color="Cas"),
                x=heatmap_hausse.columns,
                y=heatmap_hausse.index,
                color_continuous_scale=["#ffebee", "#ffcdd2", "#ef9a9a", "#e57373", "#ef5350", "#f44336", "#d32f2f"],
                aspect="auto"
            )
            fig_heat_hausse.update_xaxes(side="bottom", tickangle=-45)
            fig_heat_hausse.update_layout(height=400)
            st.plotly_chart(fig_heat_hausse, use_container_width=True)
        
        with col_heat2:
            st.markdown("**🟢 Top 10 - Baisses prédites**")
            heatmap_baisse = heatmap_data.loc[heatmap_data.index.isin(top_10_baisse)]
            
            fig_heat_baisse = px.imshow(
                heatmap_baisse,
                labels=dict(x="Semaine", y="Aire", color="Cas"),
                x=heatmap_baisse.columns,
                y=heatmap_baisse.index,
                color_continuous_scale=["#e8f5e9", "#c8e6c9", "#a5d6a7", "#81c784", "#66bb6a", "#4caf50", "#2e7d32"],
                aspect="auto"
            )
            fig_heat_baisse.update_xaxes(side="bottom", tickangle=-45)
            fig_heat_baisse.update_layout(height=400)
            st.plotly_chart(fig_heat_baisse, use_container_width=True)
        
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
            aire_name = row.get('health_area', 'Aire inconnue')
            variation_pct = row.get('Variation_Pct', 0)
            moy_historique = row.get('Moyenne_Historique', 0)
            cas_pred_moy = row.get('Cas_Predits_Moyen', 0)
            cas_pred_max = row.get('Cas_Predits_Max', 0)
            semaine_pic = row.get('Semaine_Pic', 'N/A')
            categorie = row.get('Categorie_Variation', 'N/A')
            
            if variation_pct >= seuil_hausse:
                bg_color = '#ffebee'
                var_color = '#d32f2f'
            elif variation_pct <= -seuil_baisse:
                bg_color = '#e8f5e9'
                var_color = '#2e7d32'
            else:
                bg_color = '#f5f5f5'
                var_color = '#000'
            
            popup_html = f"""
            <div style="font-family: Arial; width: 360px;">
                <h3 style="color: #1976d2; border-bottom: 2px solid #1976d2;">
                    {aire_name}
                </h3>
                <div style="background-color: {bg_color}; padding: 10px; margin: 10px 0; border-radius: 5px;">
                    <h4 style="margin: 0;">🔮 Prédictions</h4>
                    <table style="width: 100%; margin-top: 5px;">
                        <tr><td><b>Moyenne historique :</b></td><td style="text-align: right;">
                            {moy_historique:.1f} cas/sem
                        </td></tr>
                        <tr><td><b>Moyenne prédite :</b></td><td style="text-align: right;">
                            {cas_pred_moy:.1f} cas/sem
                        </td></tr>
                        <tr><td><b>Variation :</b></td><td style="text-align: right; font-size: 18px; color: {var_color};">
                            <b>{variation_pct:+.1f}%</b>
                        </td></tr>
                        <tr><td>Tendance :</td><td style="text-align: right;">
                            <b>{categorie}</b>
                        </td></tr>
                        <tr><td>Semaine du pic :</td><td style="text-align: right;">
                            {semaine_pic}
                        </td></tr>
                        <tr><td>Pic maximal :</td><td style="text-align: right;">
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
                    f"<b>{aire_name}</b><br>Variation : {variation_pct:+.1f}%",
                    sticky=True
                ),
                popup=folium.Popup(popup_html, max_width=400)
            ).add_to(m_pred)
        
        st_folium(m_pred, width=1400, height=650)
        
        st.header("💾 Export des Résultats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_pred = risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Synthèse prédictions (CSV)",
                data=csv_pred,
                file_name=f"predictions_synthese_{pays_selectionne if pays_selectionne else 'pays'}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_detail = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📅 Détail par semaine (CSV)",
                data=csv_detail,
                file_name=f"predictions_detail_{pays_selectionne if pays_selectionne else 'pays'}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            geojson_str = sa_gdf_pred.to_json()
            st.download_button(
                label="🗺️ Carte prédictions (GeoJSON)",
                data=geojson_str,
                file_name=f"carte_predictions_{pays_selectionne if pays_selectionne else 'pays'}_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/json"
            )
        
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
            file_name=f"rapport_complet_{pays_selectionne if pays_selectionne else 'pays'}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.header("💡 Recommandations Opérationnelles")
        
        aires_critiques_hausse = risk_df[risk_df["Variation_Pct"] >= seuil_hausse]["Aire_Sante"].tolist()
        aires_amelioration = risk_df[risk_df["Variation_Pct"] <= -seuil_baisse]["Aire_Sante"].tolist()
        
        if aires_critiques_hausse:
            st.error(f"🚨 **{len(aires_critiques_hausse)} aire(s) à risque CRITIQUE (hausse ≥{seuil_hausse}%)**")
            
            for i, aire in enumerate(aires_critiques_hausse[:5], 1):
                aire_info = risk_df[risk_df["Aire_Sante"] == aire].iloc[0]
                st.write(f"{i}. **{aire}** : {aire_info['Variation_Pct']:+.1f}% - Pic S{aire_info['Semaine_Pic']} ({int(aire_info['Cas_Predits_Max'])} cas)")
            
            if len(aires_critiques_hausse) > 5:
                st.caption(f"... et {len(aires_critiques_hausse)-5} autre(s)")
            
            st.write("")
            st.write("**Actions prioritaires recommandées :**")
            st.write("✅ Renforcer la surveillance épidémiologique hebdomadaire dans ces aires")
            st.write("✅ Organiser des campagnes de vaccination de rattrapage urgentes")
            st.write("✅ Prépositionner les stocks de vaccins et intrants médicaux")
            st.write("✅ Sensibiliser les communautés aux signes d'alerte de la rougeole")
            st.write("✅ Coordonner avec les partenaires (OMS, UNICEF, MSF)")
            st.write("✅ Préparer les structures de santé à une augmentation de cas")
        
        if aires_amelioration:
            st.success(f"✓ **{len(aires_amelioration)} aire(s) montrent une amélioration (baisse ≥{seuil_baisse}%)**")
            st.write("**Bonnes pratiques à capitaliser :**")
            st.write("• Documenter les interventions ayant conduit à cette baisse")
            st.write("• Partager les leçons apprises avec les autres aires")
            st.write("• Maintenir la vigilance pour éviter une résurgence")
        
        if not aires_critiques_hausse and not aires_amelioration:
            st.info("ℹ️ **Situation stable** - Aucune variation significative détectée")
            st.write("• Maintenir la surveillance de routine")
            st.write("• Continuer les activités de vaccination selon le calendrier")
        
        st.markdown("---")
        st.caption(f"""
**Méthodologie de prédiction :**
Modèle : {modele_choisi} | Score R² (validation croisée) : {cv_mean:.3f} (±{cv_std:.3f}) |
Variables : {len(feature_cols)} features (historique 4 semaines, démographie, urbanisation, climat, vaccination) |
Période : S{derniere_semaine_epi+1} à S{min(derniere_semaine_epi+n_weeks_pred, 52)} ({n_weeks_pred} semaines) |
Seuils : Baisse ≥{seuil_baisse}%, Hausse ≥{seuil_hausse}%, Alerte ≥{seuil_alerte_epidemique} cas/sem
        """)

else:
    st.info("👆 Cliquez sur le bouton ci-dessus pour lancer la modélisation prédictive")
    st.markdown("""

    """)

st.markdown("---")
st.caption(f"""
Dashboard de Surveillance Rougeole - Version 3.0 (Améliorée) |
Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} |
Pays : {pays_selectionne if pays_selectionne else 'Non spécifié'} |
Période : {start_date} - {end_date} |
Nombre d'aires : {len(sa_gdf)} |
Cas analysés : {len(df):,} |
Mode : {mode_demo}
""")
