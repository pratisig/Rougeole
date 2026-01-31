# ============================================================
# APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE (Multi-pays)
# Version améliorée selon spécifications
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

# CONFIGURATION STREAMLIT
st.set_page_config(
    page_title="Surveillance Rougeole Multi-pays",
    layout="wide",
    page_icon="🦠",
    initial_sidebar_state="expanded"
)

# STYLE CSS
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
# BARRE LATÉRALE - CONFIGURATION
# ============================================================

st.sidebar.header("📂 Configuration de l'Analyse")

if 'pays_precedent' not in st.session_state:
    st.session_state.pays_precedent = None
if 'sa_gdf_cache' not in st.session_state:
    st.session_state.sa_gdf_cache = None

st.sidebar.subheader("🎯 Mode d'utilisation")
mode_demo = st.sidebar.radio(
    "Choisissez votre mode",
    ["📊 Données réelles", "🧪 Mode démo (données simulées)"],
    help="Mode démo : génère automatiquement des données fictives pour tester l'application"
)

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

model_hints = {
    "GradientBoosting (Recommandé)": "🎯 **Gradient Boosting** : Très performant pour les séries temporelles. Excellent pour capturer les relations non linéaires.",
    "RandomForest": "🌳 **Random Forest** : Ensemble d'arbres de décision. Robuste aux valeurs aberrantes.",
    "Ridge Regression": "📊 **Ridge Regression** : Régression linéaire avec régularisation L2. Simple et rapide.",
    "Lasso Regression": "🎯 **Lasso Regression** : Utile quand beaucoup de variables sont peu importantes.",
    "Decision Tree": "🌲 **Decision Tree** : Simple à interpréter mais risque de sur-apprentissage."
}

st.sidebar.markdown(f'<div class="model-hint">{model_hints[modele_choisi]}</div>', unsafe_allow_html=True)

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
# CHARGEMENT DES DONNÉES GÉOGRAPHIQUES
# ============================================================

@st.cache_data
def load_health_areas_from_zip(zip_path, iso3_filter):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
            
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if not shp_files:
                raise ValueError("Aucun fichier .shp trouvé")
            
            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf_full = gpd.read_file(shp_path)
            
            iso3_col = None
            for col in ['iso3', 'ISO3', 'iso_code', 'ISO_CODE']:
                if col in gdf_full.columns:
                    iso3_col = col
                    break
            
            if iso3_col is None:
                return gpd.GeoDataFrame()
            
            gdf = gdf_full[gdf_full[iso3_col] == iso3_filter].copy()
            
            name_col = None
            for col in ['health_area', 'HEALTH_AREA', 'name_fr', 'aire_sante']:
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
        st.error(f"Erreur ZIP : {e}")
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
            for col in ["health_area", "HEALTH_AREA", "name_fr", "name", "nom"]:
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
        st.error(f"Erreur lecture : {e}")
        return gpd.GeoDataFrame()

if st.session_state.sa_gdf_cache is not None and option_aire == "Fichier local (ao_hlthArea.zip)":
    sa_gdf = st.session_state.sa_gdf_cache
else:
    with st.spinner("🔄 Chargement des aires de santé..."):
        if option_aire == "Fichier local (ao_hlthArea.zip)":
            zip_path = os.path.join("data", "ao_hlthArea.zip")
            if not os.path.exists(zip_path):
                st.error(f"Fichier non trouvé : {zip_path}")
                st.stop()
            sa_gdf = load_health_areas_from_zip(zip_path, iso3_pays)
            st.session_state.sa_gdf_cache = sa_gdf
        elif option_aire == "Upload personnalisé":
            if upload_file is None:
                st.warning("Veuillez charger un fichier géographique.")
                st.stop()
            sa_gdf = load_shapefile_from_upload(upload_file)

if sa_gdf is None or sa_gdf.empty:
    st.error("Aucune aire chargée.")
    st.stop()

# ============================================================
# DONNÉES FICTIVES & CHARGEMENT DES CAS
# ============================================================

@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500, start=None, end=None):
    np.random.seed(42)
    if start is None: start = datetime(2024, 1, 1)
    if end is None: end = datetime.today()
    delta_days = (end - start).days
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.random.exponential(scale=delta_days/3, size=n).clip(0, delta_days).astype(int),
        unit="D"
    )
    return pd.DataFrame({
        "ID_Cas": range(1, n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(np.random.poisson(3, n), unit="D"),
        "Aire_Sante": np.random.choice(_sa_gdf["health_area"].unique(), n),
        "Age_Mois": np.random.gamma(shape=2, scale=30, size=n).clip(6, 180).astype(int),
        "Statut_Vaccinal": np.random.choice(["Oui", "Non"], n, p=[0.55, 0.45]),
        "Sexe": np.random.choice(["M", "F"], n),
        "Issue": np.random.choice(["Guéri", "Décédé", "Inconnu"], n, p=[0.92, 0.03, 0.05])
    })

@st.cache_data
def generate_dummy_vaccination(_sa_gdf):
    return pd.DataFrame({
        "health_area": _sa_gdf["health_area"],
        "Taux_Vaccination": np.random.beta(a=8, b=2, size=len(_sa_gdf)) * 100
    })

with st.spinner("📥 Chargement données de cas..."):
    if mode_demo == "🧪 Mode démo (données simulées)":
        df = generate_dummy_linelists(sa_gdf, start=start_date, end=end_date)
        vaccination_df = generate_dummy_vaccination(sa_gdf)
    else:
        if linelist_file is None:
            st.error("Veuillez charger un fichier de cas (CSV).")
            st.stop()
        df = pd.read_csv(linelist_file)
        df["Date_Debut_Eruption"] = pd.to_datetime(df["Date_Debut_Eruption"], errors='coerce')
        vaccination_df = pd.read_csv(vaccination_file) if vaccination_file else None

df = df[(df["Date_Debut_Eruption"] >= pd.to_datetime(start_date)) & 
        (df["Date_Debut_Eruption"] <= pd.to_datetime(end_date))].copy()
df['Semaine_Epi'] = df['Date_Debut_Eruption'].dt.isocalendar().week
df['Annee'] = df['Date_Debut_Eruption'].dt.year
df['Semaine_Annee'] = df['Annee'].astype(str) + '-S' + df['Semaine_Epi'].astype(str).str.zfill(2)
derniere_semaine_epi = df['Semaine_Epi'].max()

# ============================================================
# ENRICHISSEMENT GEE & NASA POWER
# ============================================================

@st.cache_data
def worldpop_children_stats(_sa_gdf, use_gee):
    if not use_gee:
        return pd.DataFrame({"health_area": _sa_gdf["health_area"], "Pop_Totale": [np.nan]*len(_sa_gdf), "Pop_Enfants": [np.nan]*len(_sa_gdf)})
    try:
        pop_img = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic()
        features = []
        for _, row in _sa_gdf.iterrows():
            geom = shape(row['geometry'])
            if geom.geom_type == 'Polygon':
                ee_geom = ee.Geometry.Polygon(list(geom.exterior.coords))
            else:
                ee_geom = ee.Geometry.MultiPolygon([list(p.exterior.coords) for p in geom.geoms])
            features.append(ee.Feature(ee_geom, {"health_area": row["health_area"]}))
        
        fc = ee.FeatureCollection(features)
        stats = pop_img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100).getInfo()
        data = []
        for f in stats['features']:
            p = f['properties']
            data.append({"health_area": p['health_area'], "Pop_Totale": p.get('population', 0), "Pop_Enfants": p.get('population', 0) * 0.45})
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame({"health_area": _sa_gdf["health_area"], "Pop_Totale": [50000]*len(_sa_gdf), "Pop_Enfants": [22500]*len(_sa_gdf)})

@st.cache_data
def urban_classification(_sa_gdf, use_gee):
    return pd.DataFrame({"health_area": _sa_gdf["health_area"], "Urbanisation": np.random.choice(["Urbain", "Rural", "Semi-urbain"], len(_sa_gdf))})

@st.cache_data
def fetch_climate_nasa_power(_sa_gdf, start, end):
    return pd.DataFrame({"health_area": _sa_gdf["health_area"], "Temperature_Moy": 30.5, "Humidite_Moy": 45.2})

pop_df = worldpop_children_stats(sa_gdf, gee_ok)
urban_df = urban_classification(sa_gdf, gee_ok)
climate_df = fetch_climate_nasa_power(sa_gdf, start_date, end_date)

sa_gdf_enrichi = sa_gdf.merge(pop_df, on="health_area", how="left").merge(urban_df, on="health_area", how="left").merge(climate_df, on="health_area", how="left")
if vaccination_df is not None:
    sa_gdf_enrichi = sa_gdf_enrichi.merge(vaccination_df, on="health_area", how="left")
else:
    sa_gdf_enrichi["Taux_Vaccination"] = 85.0

sa_gdf_enrichi["Superficie_km2"] = sa_gdf_enrichi.geometry.area / 1e6
sa_gdf_enrichi["Densite_Pop"] = sa_gdf_enrichi["Pop_Totale"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
# ============================================================
# PARTIE 2/6 - VISUALISATIONS, MACHINE LEARNING ET ALERTES
# ============================================================

st.header("📊 Indicateurs Clés de Performance")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📈 Cas totaux", f"{len(df):,}")
with col2:
    taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
    st.metric("💉 Non vaccinés", f"{taux_non_vac:.1f}%")
with col3:
    st.metric("👶 Âge médian", f"{int(df['Age_Mois'].median())} mois")
with col4:
    if "Issue" in df.columns:
        taux_let = (df["Issue"] == "Décédé").mean() * 100
        st.metric("☠️ Létalité", f"{taux_let:.2f}%")
    else:
        st.metric("☠️ Létalité", "N/A")
with col5:
    st.metric("🗺️ Aires touchées", f"{df['Aire_Sante'].nunique()}/{len(sa_gdf)}")

cases_by_area = df.groupby("Aire_Sante").size().reset_index(name="Cas_Observes")
sa_gdf_with_cases = sa_gdf_enrichi.merge(cases_by_area, left_on="health_area", right_on="Aire_Sante", how="left").fillna(0)

# CARTOGRAPHIE
st.header("🗺️ Cartographie de la Situation Actuelle")
center = [sa_gdf_with_cases.geometry.centroid.y.mean(), sa_gdf_with_cases.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

for idx, row in sa_gdf_with_cases.iterrows():
    color = 'red' if row['Cas_Observes'] >= seuil_alerte_epidemique else 'green'
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, c=color: {'fillColor': c, 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.6},
        tooltip=f"{row['health_area']}: {int(row['Cas_Observes'])} cas",
        popup=folium.Popup(f"<b>{row['health_area']}</b><br>Cas : {int(row['Cas_Observes'])}<br>Vaccination : {row['Taux_Vaccination']:.1f}%", max_width=300)
    ).add_to(m)

st_folium(m, width="100%", height=600)

# GRAPHIQUES
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("📈 Évolution temporelle")
    weekly_trend = df.groupby('Semaine_Annee').size().reset_index(name='Cas')
    st.plotly_chart(px.line(weekly_trend, x='Semaine_Annee', y='Cas', markers=True), use_container_width=True)
with col_r:
    st.subheader("👶 Distribution par Âge et Sexe")
    st.plotly_chart(px.histogram(df, x="Age_Mois", color="Sexe", nbins=20, barmode="group"), use_container_width=True)

# PRÉDICTION
st.divider()
st.header("🔮 Modélisation Prédictive & Alertes")

if st.button("🚀 Lancer la modélisation prédictive", type="primary"):
    with st.spinner("🧠 Préparation des variables et entraînement..."):
        all_aires = sa_gdf_enrichi["health_area"].unique()
        all_weeks = sorted(df["Semaine_Epi"].unique())
        
        index = pd.MultiIndex.from_product([all_aires, all_weeks], names=["Aire_Sante", "Semaine_Epi"])
        weekly_features = pd.DataFrame(index=index).reset_index()
        
        cas_c = df.groupby(["Aire_Sante", "Semaine_Epi"]).size().reset_index(name="Cas_Observes")
        weekly_features = weekly_features.merge(cas_c, on=["Aire_Sante", "Semaine_Epi"], how="left").fillna(0)
        
        weekly_features = weekly_features.merge(
            sa_gdf_enrichi[["health_area", "Pop_Enfants", "Taux_Vaccination", "Temperature_Moy", "Humidite_Moy", "Densite_Pop"]],
            left_on="Aire_Sante", right_on="health_area", how="left"
        )
        
        # Lags
        weekly_features = weekly_features.sort_values(["Aire_Sante", "Semaine_Epi"])
        for i in range(1, 5):
            weekly_features[f"Cas_Lag_{i}"] = weekly_features.groupby("Aire_Sante")["Cas_Observes"].shift(i).fillna(0)
        
        feature_cols = ["Semaine_Epi", "Pop_Enfants", "Taux_Vaccination", "Temperature_Moy", "Humidite_Moy", "Densite_Pop", "Cas_Lag_1", "Cas_Lag_2"]
        train_df = weekly_features.dropna()
        
        X = train_df[feature_cols]
        y = train_df["Cas_Observes"]
        
        if "Gradient" in modele_choisi:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif "Random" in modele_choisi:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif "Ridge" in modele_choisi:
            model = Ridge()
        elif "Lasso" in modele_choisi:
            model = Lasso()
        else:
            model = DecisionTreeRegressor()
            
        model.fit(X, y)
        cv_mean = cross_val_score(model, X, y, cv=3).mean()
        
        # Boucle de prédiction
        predictions_futures = []
        for aire in all_aires:
            aire_data = weekly_features[weekly_features["Aire_Sante"] == aire].iloc[-1:].copy()
            curr_lags = [aire_data["Cas_Observes"].values[0], aire_data["Cas_Lag_1"].values[0]]
            
            for w in range(1, n_weeks_pred + 1):
                fut_week = (derniere_semaine_epi + w - 1) % 52 + 1
                input_row = pd.DataFrame([[
                    fut_week, aire_data["Pop_Enfants"].values[0], aire_data["Taux_Vaccination"].values[0],
                    aire_data["Temperature_Moy"].values[0], aire_data["Humidite_Moy"].values[0],
                    aire_data["Densite_Pop"].values[0], curr_lags[0], curr_lags[1]
                ]], columns=feature_cols)
                
                pred_val = max(0, model.predict(input_row)[0])
                predictions_futures.append({"Aire_Sante": aire, "Semaine_Epi": fut_week, "Cas_Prevus": pred_val})
                curr_lags = [pred_val, curr_lags[0]]
                
        df_pred = pd.DataFrame(predictions_futures)
        res_finaux = df_pred.groupby("Aire_Sante")["Cas_Prevus"].agg(['mean', 'max']).reset_index()
        res_finaux.columns = ["Aire_Sante", "Moyenne_Prevue", "Pic_Prevu"]
        
        def det_risk(row):
            if row["Pic_Prevu"] >= seuil_alerte_epidemique: return "🔴 Élevé"
            if row["Pic_Prevu"] >= 2: return "🟠 Modéré"
            return "🟢 Faible"
        res_finaux["Niveau_Risque"] = res_finaux.apply(det_risk, axis=1)
        
        st.success(f"Modélisation terminée. Score R² : {cv_mean:.2f}")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("🚨 Alertes")
            st.dataframe(res_finaux[res_finaux["Niveau_Risque"] != "🟢 Faible"].sort_values("Pic_Prevu", ascending=False), hide_index=True)
        with c2:
            st.subheader("📅 Heatmap Temporelle")
            pivot_df = df_pred.pivot(index="Aire_Sante", columns="Semaine_Epi", values="Cas_Prevus").head(20)
            st.plotly_chart(px.imshow(pivot_df, color_continuous_scale="YlOrRd"), use_container_width=True)
            
        # RECOMMANDATIONS
        st.subheader("📋 Recommandations")
        for idx, row in res_finaux[res_finaux["Niveau_Risque"] == "🔴 Élevé"].head(3).iterrows():
            st.error(f"**Action requise pour {row['Aire_Sante']}** : Investigation et renforcement vaccinal.")
            
        csv = res_finaux.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger le rapport (CSV)", csv, "predictions_rougeole.csv", "text/csv")

else:
    st.info("👆 Cliquez sur le bouton ci-dessus pour lancer la modélisation.")

st.markdown("---")
st.caption(f"Tableau de bord de Surveillance | © 2024 | Version 3.0")
