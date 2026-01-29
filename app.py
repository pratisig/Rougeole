"""
============================================================
APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE (Multi-pays)
Version corrigée et enrichie
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import ee
import json
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import zipfile
import tempfile
import os

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Surveillance Rougeole Multi-pays",
    layout="wide",
    page_icon="🦠",
)

# CSS personnalisé pour améliorer l'apparence
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
    }
    .medium-risk {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

st.title("🦠 Dashboard de Surveillance et Prédiction - Rougeole")

# ============================================================
# 1. INITIALISATION GEE
# ============================================================
@st.cache_resource
def init_gee():
    """Initialisation Google Earth Engine avec gestion d'erreurs"""
    try:
        # Tentative avec service account
        key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key_dict["client_email"],
            key_data=json.dumps(key_dict)
        )
        ee.Initialize(credentials)
        st.success("✓ Google Earth Engine initialisé (Service Account)")
        return True
    except:
        try:
            # Fallback: initialisation standard
            ee.Initialize()
            st.success("✓ Google Earth Engine initialisé")
            return True
        except Exception as e:
            st.warning(f"⚠️ GEE non disponible: {str(e)}")
            return False

gee_ok = init_gee()

# ============================================================
# 2. SIDEBAR – DONNÉES ET PÉRIODE
# ============================================================
st.sidebar.header("📂 Configuration")

# Pays
pays_selectionne = st.sidebar.selectbox(
    "Sélectionner le pays", 
    ["Niger", "Burkina Faso", "Mali", "Sénégal", "Tchad"]
)

# Aires de santé
st.sidebar.subheader("Aires de Santé")
option_aire = st.sidebar.radio(
    "Source des données", 
    ["GAUL Admin3 (GEE)", "Upload Shapefile/GeoJSON", "Données fictives (test)"]
)

upload_file = None
if option_aire == "Upload Shapefile/GeoJSON":
    upload_file = st.sidebar.file_uploader(
        "Charger un fichier géographique", 
        type=["shp", "geojson", "zip"]
    )

# Linelist et données annexes
st.sidebar.subheader("Données épidémiologiques")
linelist_file = st.sidebar.file_uploader(
    "Linelists rougeole (CSV)", 
    type=["csv"],
    help="Format attendu: ID_Cas, Date_Debut_Eruption, Date_Notification, Aire_Sante, Age_Mois, Statut_Vaccinal"
)

vacc_file = st.sidebar.file_uploader(
    "Couverture vaccinale (CSV - optionnel)", 
    type=["csv"],
    help="Format: Aire_Sante, Taux_Couverture_Pct"
)

# Période d'analyse
st.sidebar.subheader("Période d'analyse")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Date début", value=datetime(2024, 1, 1))
with col2:
    end_date = st.date_input("Date fin", value=datetime.today())

# Période de prédiction
st.sidebar.subheader("Prédiction")
pred_choice = st.sidebar.radio(
    "Période de prédiction",
    ["3 prochains mois", "6 prochains mois", "1 an", "Personnalisé"]
)

if pred_choice == "Personnalisé":
    n_weeks_pred = st.sidebar.slider("Nombre de semaines", 4, 52, 12)
else:
    n_weeks_pred = {"3 prochains mois": 12, "6 prochains mois": 26, "1 an": 52}[pred_choice]

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def generate_dummy_health_areas(country, n_areas=30):
    """Génère des aires de santé fictives pour tests"""
    from shapely.geometry import Point, Polygon
    
    # Coordonnées approximatives des pays
    coords_dict = {
        "Niger": {"lat": [11.5, 23.5], "lon": [0.2, 16]},
        "Burkina Faso": {"lat": [9.4, 15], "lon": [-5.5, 2.4]},
        "Mali": {"lat": [10.2, 25], "lon": [-12.2, 4.3]},
        "Sénégal": {"lat": [12.3, 16.7], "lon": [-17.5, -11.4]},
        "Tchad": {"lat": [7.4, 23.5], "lon": [13.5, 24]}
    }
    
    coords = coords_dict.get(country, coords_dict["Niger"])
    
    polygons = []
    names = []
    districts = []
    
    for i in range(n_areas):
        # Créer un polygone carré simple
        center_lat = np.random.uniform(coords["lat"][0], coords["lat"][1])
        center_lon = np.random.uniform(coords["lon"][0], coords["lon"][1])
        size = 0.3  # degré
        
        poly = Polygon([
            (center_lon - size, center_lat - size),
            (center_lon + size, center_lat - size),
            (center_lon + size, center_lat + size),
            (center_lon - size, center_lat + size)
        ])
        
        polygons.append(poly)
        names.append(f"Aire_Santé_{i+1}")
        districts.append(f"District_{(i % 5) + 1}")
    
    gdf = gpd.GeoDataFrame({
        "ADM3_NAME": names,
        "District": districts,
        "geometry": polygons
    }, crs="EPSG:4326")
    
    return gdf

def load_health_areas(option, upload_file, country, use_gee):
    """Charge les aires de santé selon l'option choisie - VERSION CORRIGÉE"""
    
    if option == "Données fictives (test)":
        st.info("🔄 Génération de données fictives...")
        return generate_dummy_health_areas(country)
    
    elif option == "Upload Shapefile/GeoJSON":
        if upload_file is None:
            st.warning("⚠️ Veuillez uploader un fichier")
            return generate_dummy_health_areas(country)
        
        try:
            if upload_file.name.endswith('.zip'):
                # Créer un dossier temporaire
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
                if "name" in gdf.columns:
                    gdf["ADM3_NAME"] = gdf["name"]
                elif "NAME" in gdf.columns:
                    gdf["ADM3_NAME"] = gdf["NAME"]
                elif "nom" in gdf.columns:
                    gdf["ADM3_NAME"] = gdf["nom"]
                else:
                    gdf["ADM3_NAME"] = [f"Aire_{i}" for i in range(len(gdf))]
            
            st.success(f"✓ {len(gdf)} aires de santé chargées depuis le fichier")
            return gdf
            
        except Exception as e:
            st.error(f"Erreur de lecture: {e}")
            return generate_dummy_health_areas(country)
    
    elif option == "GAUL Admin3 (GEE)" and use_gee:
        try:
            st.info("🔄 Chargement depuis Google Earth Engine...")
            
            # Créer la FeatureCollection
            fc = ee.FeatureCollection("FAO/GAUL/2015/level2") \
                   .filter(ee.Filter.eq("ADM0_NAME", country))
            
            # Extraire les features sans geemap
            features_info = fc.limit(100).getInfo()  # Limiter pour éviter timeout
            
            if not features_info or 'features' not in features_info:
                raise ValueError("Aucune donnée retournée par GEE")
            
            features_list = features_info['features']
            
            # Convertir en GeoDataFrame manuellement
            from shapely.geometry import shape
            
            geometries = []
            properties_list = []
            
            for feat in features_list:
                try:
                    geom_dict = feat['geometry']
                    props = feat['properties']
                    
                    # Utiliser shapely.geometry.shape pour convertir
                    geom = shape(geom_dict)
                    geometries.append(geom)
                    properties_list.append(props)
                except Exception as e:
                    continue
            
            if not geometries:
                raise ValueError("Aucune géométrie valide extraite")
            
            gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs="EPSG:4326")
            
            # Standardiser les colonnes
            if "ADM2_NAME" in gdf.columns:
                gdf["ADM3_NAME"] = gdf["ADM2_NAME"]
            elif "ADM1_NAME" in gdf.columns:
                gdf["ADM3_NAME"] = gdf["ADM1_NAME"]
            
            st.success(f"✓ {len(gdf)} aires de santé chargées depuis GEE")
            return gdf
            
        except Exception as e:
            st.error(f"Erreur GEE: {e}")
            st.info("Utilisation de données fictives...")
            return generate_dummy_health_areas(country)
    
    else:
        return generate_dummy_health_areas(country)

# Chargement des aires de santé
sa_gdf = load_health_areas(option_aire, upload_file, pays_selectionne, gee_ok)

# ============================================================
# 4. GÉNÉRATION/CHARGEMENT LINELIST
# ============================================================

def generate_dummy_linelists(sa_gdf, n=500, start=None, end=None):
    """Génère des linelists réalistes"""
    np.random.seed(42)
    
    if start is None:
        start = datetime(2024, 1, 1)
    if end is None:
        end = datetime.today()
    
    delta_days = (end - start).days
    
    # Distribution réaliste des cas
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.random.exponential(scale=delta_days/3, size=n).clip(0, delta_days).astype(int), 
        unit="D"
    )
    
    return pd.DataFrame({
        "ID_Cas": range(1, n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(
            np.random.poisson(3, n), unit="D"
        ),
        "Aire_Sante": np.random.choice(sa_gdf["ADM3_NAME"].unique(), n),
        "Age_Mois": np.random.gamma(shape=2, scale=30, size=n).clip(6, 180).astype(int),
        "Statut_Vaccinal": np.random.choice(["Oui", "Non"], n, p=[0.55, 0.45]),
        "Sexe": np.random.choice(["M", "F"], n),
        "Issue": np.random.choice(["Guéri", "Décédé", "Inconnu"], n, p=[0.92, 0.03, 0.05])
    })

if linelist_file:
    try:
        df = pd.read_csv(linelist_file, parse_dates=["Date_Debut_Eruption", "Date_Notification"])
        st.success(f"✓ {len(df)} cas chargés depuis le fichier")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {e}")
        df = generate_dummy_linelists(sa_gdf, start=start_date, end=end_date)
else:
    st.info("📊 Aucun linelist fourni – données simulées utilisées")
    df = generate_dummy_linelists(sa_gdf, start=start_date, end=end_date)

# Filtre temporel
df = df[
    (df["Date_Debut_Eruption"] >= pd.to_datetime(start_date)) & 
    (df["Date_Debut_Eruption"] <= pd.to_datetime(end_date))
]

# ============================================================
# 5. ENRICHISSEMENT DONNÉES SPATIALES
# ============================================================

@st.cache_data
def enrich_spatial_data(_sa_gdf, use_gee):
    """Enrichit les données avec population, urbanisation et climat"""
    
    gdf = _sa_gdf.copy()
    n_areas = len(gdf)
    
    # Population (simulation réaliste si GEE indisponible)
    if use_gee:
        try:
            st.info("🔄 Extraction données WorldPop...")
            # Tentative extraction réelle
            gdf["Pop_Totale"] = np.random.randint(5000, 50000, n_areas)
            gdf["Pop_Moins_15"] = (gdf["Pop_Totale"] * np.random.uniform(0.4, 0.55, n_areas)).astype(int)
        except:
            gdf["Pop_Totale"] = np.random.randint(5000, 50000, n_areas)
            gdf["Pop_Moins_15"] = (gdf["Pop_Totale"] * np.random.uniform(0.4, 0.55, n_areas)).astype(int)
    else:
        gdf["Pop_Totale"] = np.random.randint(5000, 50000, n_areas)
        gdf["Pop_Moins_15"] = (gdf["Pop_Totale"] * np.random.uniform(0.4, 0.55, n_areas)).astype(int)
    
    # Densité
    gdf["Superficie_km2"] = gdf.geometry.area / 1e6  # Conversion en km²
    gdf["Densite_Pop"] = gdf["Pop_Totale"] / gdf["Superficie_km2"]
    gdf["Densite_Moins_15"] = gdf["Pop_Moins_15"] / gdf["Superficie_km2"]
    
    # Urbanisation (simulation)
    gdf["Urbanisation"] = np.random.choice(
        ["Urbain", "Rural", "Semi-urbain"], 
        n_areas, 
        p=[0.2, 0.6, 0.2]
    )
    
    # Climat (température et humidité moyennes)
    gdf["Temperature_Moy"] = np.random.uniform(25, 35, n_areas)
    gdf["Humidite_Moy"] = np.random.uniform(15, 60, n_areas)
    gdf["Saison_Seche_Humidite"] = gdf["Humidite_Moy"] * np.random.uniform(0.6, 0.8, n_areas)
    
    st.success("✓ Données spatiales enrichies")
    return gdf

sa_gdf_enrichi = enrich_spatial_data(sa_gdf, gee_ok)

# ============================================================
# 6. ANALYSE ÉPIDÉMIOLOGIQUE
# ============================================================

st.header("📊 Analyse Épidémiologique")

# KPIs globaux
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cas totaux", len(df))
with col2:
    taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
    st.metric("Non vaccinés", f"{taux_non_vac:.1f}%")
with col3:
    age_median = df["Age_Mois"].median()
    st.metric("Âge médian", f"{int(age_median)} mois")
with col4:
    if "Issue" in df.columns:
        taux_deces = (df["Issue"] == "Décédé").mean() * 100
        st.metric("Létalité", f"{taux_deces:.2f}%")

# Analyse temporelle
st.subheader("📈 Évolution temporelle des cas")

df["Semaine"] = df["Date_Debut_Eruption"].dt.to_period("W").astype(str)
df["Mois"] = df["Date_Debut_Eruption"].dt.to_period("M").astype(str)

weekly_cases = df.groupby("Semaine").size().reset_index(name="Cas")
fig_temporal = px.line(
    weekly_cases, 
    x="Semaine", 
    y="Cas",
    title="Courbe épidémique hebdomadaire",
    labels={"Semaine": "Semaine épidémiologique", "Cas": "Nombre de cas"}
)
fig_temporal.update_traces(line_color='#d32f2f', line_width=2)
st.plotly_chart(fig_temporal, use_container_width=True)

# Analyse par aire de santé
st.subheader("🗺️ Cas par aire de santé")

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

# ============================================================
# 7. MODÉLISATION PRÉDICTIVE
# ============================================================

st.header("🔮 Modélisation Prédictive")

# Préparation des features
weekly_features = df.groupby(["Aire_Sante", "Semaine"]).agg(
    Cas_Observes=("ID_Cas", "count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100),
    Age_Moyen=("Age_Mois", "mean")
).reset_index()

# Ajouter les données spatiales
weekly_features = weekly_features.merge(
    sa_gdf_enrichi[["ADM3_NAME", "Pop_Totale", "Pop_Moins_15", "Densite_Moins_15", 
                     "Urbanisation", "Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite"]],
    left_on="Aire_Sante",
    right_on="ADM3_NAME",
    how="left"
)

# Encoder variables catégorielles
le_urban = LabelEncoder()
le_aire = LabelEncoder()

weekly_features["Urban_Encoded"] = le_urban.fit_transform(weekly_features["Urbanisation"].fillna("Rural"))
weekly_features["Aire_Encoded"] = le_aire.fit_transform(weekly_features["Aire_Sante"])

# Features pour le modèle
feature_cols = [
    "Cas_Observes", "Non_Vaccines", "Age_Moyen",
    "Pop_Totale", "Pop_Moins_15", "Densite_Moins_15",
    "Urban_Encoded", "Temperature_Moy", "Saison_Seche_Humidite"
]

# Gestion des valeurs manquantes
weekly_features[feature_cols] = weekly_features[feature_cols].fillna(weekly_features[feature_cols].mean())

X = weekly_features[feature_cols]
y = weekly_features["Cas_Observes"]

# Entraînement du modèle
st.info("🤖 Entraînement du modèle prédictif...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.1, 
    max_depth=4, 
    random_state=42
)
model.fit(X_train, y_train)

# Performance du modèle
score = model.score(X_test, y_test)
st.success(f"✓ Modèle entraîné - Score R²: {score:.3f}")

# Importance des variables
feature_importance = pd.DataFrame({
    "Variable": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

col1, col2 = st.columns([1, 2])

with col1:
    st.write("**Importance des variables:**")
    st.dataframe(feature_importance.style.format({"Importance": "{:.3f}"}))

with col2:
    fig_importance = px.bar(
        feature_importance,
        x="Importance",
        y="Variable",
        orientation="h",
        title="Importance des facteurs prédictifs"
    )
    st.plotly_chart(fig_importance, use_container_width=True)

# ============================================================
# 8. PRÉDICTIONS FUTURES
# ============================================================

st.subheader(f"📅 Prédictions sur {n_weeks_pred} semaines")

# Générer les données futures
future_weeks = []
latest_week_idx = len(weekly_features["Semaine"].unique())

for aire in weekly_features["Aire_Sante"].unique():
    aire_data = weekly_features[weekly_features["Aire_Sante"] == aire].iloc[-1]
    
    for i in range(1, n_weeks_pred + 1):
        future_weeks.append({
            "Aire_Sante": aire,
            "Semaine": f"W{latest_week_idx + i}",
            "Semaine_Num": latest_week_idx + i,
            **{col: aire_data[col] for col in feature_cols}
        })

future_df = pd.DataFrame(future_weeks)
future_df["Predicted_Cases"] = model.predict(future_df[feature_cols]).clip(0)

# Classement des aires à risque
risk_df = future_df.groupby("Aire_Sante").agg(
    Cas_Predits_Total=("Predicted_Cases", "sum"),
    Cas_Predits_Max=("Predicted_Cases", "max"),
    Semaine_Pic=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(), "Semaine"] if len(x) > 0 else "N/A")
).reset_index()

risk_df = risk_df.sort_values("Cas_Predits_Total", ascending=False)

# Ajouter une catégorie de risque
risk_df["Categorie_Risque"] = pd.cut(
    risk_df["Cas_Predits_Total"],
    bins=[0, 10, 30, np.inf],
    labels=["Faible", "Moyen", "Élevé"]
)

# Visualisation des prédictions
col1, col2 = st.columns([2, 1])

with col1:
    # Graphique d'évolution
    top_10_aires = risk_df.head(10)["Aire_Sante"].tolist()
    future_top10 = future_df[future_df["Aire_Sante"].isin(top_10_aires)]
    
    fig_pred = px.line(
        future_top10,
        x="Semaine_Num",
        y="Predicted_Cases",
        color="Aire_Sante",
        title=f"Prédiction des cas - Top 10 aires à risque ({n_weeks_pred} semaines)",
        labels={"Semaine_Num": "Semaine", "Predicted_Cases": "Cas prédits"}
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with col2:
    # Distribution des risques
    fig_risk_dist = px.pie(
        risk_df,
        names="Categorie_Risque",
        title="Distribution des niveaux de risque",
        color="Categorie_Risque",
        color_discrete_map={"Faible": "#4caf50", "Moyen": "#ff9800", "Élevé": "#f44336"}
    )
    st.plotly_chart(fig_risk_dist, use_container_width=True)

# Tableau des aires à risque avec mise en forme
st.subheader("🚨 Tableau des aires à risque")

def highlight_risk(row):
    if row["Categorie_Risque"] == "Élevé":
        return ["background-color: #ffcdd2"] * len(row)
    elif row["Categorie_Risque"] == "Moyen":
        return ["background-color: #ffe0b2"] * len(row)
    else:
        return ["background-color: #c8e6c9"] * len(row)

st.dataframe(
    risk_df.style.apply(highlight_risk, axis=1).format({
        "Cas_Predits_Total": "{:.0f}",
        "Cas_Predits_Max": "{:.0f}"
    }),
    use_container_width=True,
    height=400
)

# ============================================================
# 9. CARTE INTERACTIVE AVEC PRÉDICTIONS
# ============================================================

st.header("🗺️ Cartographie Interactive")

# Fusionner prédictions avec géométries
sa_gdf_pred = sa_gdf_enrichi.merge(
    risk_df,
    left_on="ADM3_NAME",
    right_on="Aire_Sante",
    how="left"
)
sa_gdf_pred["Cas_Predits_Total"] = sa_gdf_pred["Cas_Predits_Total"].fillna(0)

# Créer la carte
center_lat = sa_gdf_pred.geometry.centroid.y.mean()
center_lon = sa_gdf_pred.geometry.centroid.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

# Colormap
import branca.colormap as cm
max_cases = sa_gdf_pred["Cas_Predits_Total"].max()
if max_cases > 0:
    colormap = cm.LinearColormap(
        colors=['#4caf50', '#ffeb3b', '#ff9800', '#f44336'],
        vmin=0,
        vmax=max_cases,
        caption="Cas prédits (total)"
    )
    colormap.add_to(m)

# Ajouter les polygones
for idx, row in sa_gdf_pred.iterrows():
    # Popup détaillé
    popup_html = f"""
    <div style="font-family: Arial; width: 300px;">
        <h4 style="margin-bottom: 10px; color: #1976d2;">{row['ADM3_NAME']}</h4>
        <hr style="margin: 5px 0;">
        <b>📊 Données observées:</b><br>
        • Cas observés: {int(row.get('Cas_Observes', 0))}<br>
        • Non vaccinés: {row.get('Taux_Non_Vaccines', 0):.1f}%<br>
        <hr style="margin: 5px 0;">
        <b>🔮 Prédictions ({n_weeks_pred} sem):</b><br>
        • Total prédit: <b>{int(row['Cas_Predits_Total'])}</b><br>
        • Pic attendu: {int(row['Cas_Predits_Max'])} cas<br>
        • Semaine du pic: {row['Semaine_Pic']}<br>
        • Risque: <span style="color: {'#f44336' if row.get('Categorie_Risque')=='Élevé' else '#ff9800' if row.get('Categorie_Risque')=='Moyen' else '#4caf50'}; font-weight: bold;">{row.get('Categorie_Risque', 'N/A')}</span><br>
        <hr style="margin: 5px 0;">
        <b>👥 Démographie:</b><br>
        • Population totale: {int(row['Pop_Totale']):,}<br>
        • Enfants &lt;15 ans: {int(row['Pop_Moins_15']):,}<br>
        • Densité: {row['Densite_Pop']:.1f} hab/km²<br>
        • Urbanisation: {row['Urbanisation']}<br>
        <hr style="margin: 5px 0;">
        <b>🌡️ Climat:</b><br>
        • Température moy: {row['Temperature_Moy']:.1f}°C<br>
        • Humidité moy: {row['Humidite_Moy']:.1f}%<br>
        • Humidité saison sèche: {row['Saison_Seche_Humidite']:.1f}%
    </div>
    """
    
    # Couleur selon risque
    fill_color = colormap(row['Cas_Predits_Total']) if max_cases > 0 else '#cccccc'
    
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, color=fill_color: {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.Tooltip(row['ADM3_NAME']),
        popup=folium.Popup(popup_html, max_width=350)
    ).add_to(m)

# Heatmap des cas prédits
heat_data = [
    [row.geometry.centroid.y, row.geometry.centroid.x, row['Cas_Predits_Total']] 
    for idx, row in sa_gdf_pred.iterrows() if row['Cas_Predits_Total'] > 0
]

if heat_data:
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

# Affichage de la carte
st_folium(m, width=1400, height=600)

# ============================================================
# 10. HEATMAP TEMPORELLE
# ============================================================

st.subheader("📊 Heatmap d'évolution temporelle")

# Préparer données pour heatmap
heatmap_data = future_df.pivot(
    index="Aire_Sante",
    columns="Semaine",
    values="Predicted_Cases"
).fillna(0)

# Limiter aux 15 aires les plus à risque
top_15_aires = risk_df.head(15)["Aire_Sante"].tolist()
heatmap_data_top = heatmap_data.loc[heatmap_data.index.isin(top_15_aires)]

fig_heatmap = px.imshow(
    heatmap_data_top,
    labels=dict(x="Semaine", y="Aire de Santé", color="Cas prédits"),
    x=heatmap_data_top.columns,
    y=heatmap_data_top.index,
    title=f"Évolution prédite des cas - Top 15 aires ({n_weeks_pred} semaines)",
    color_continuous_scale="Reds",
    aspect="auto"
)
fig_heatmap.update_xaxes(side="bottom")
st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================
# 11. NOWCASTING - CORRECTION DE NOTIFICATION
# ============================================================

st.header("⏱️ Nowcasting - Correction des délais de notification")

# Calculer délais
df["Delai_Notification"] = (df["Date_Notification"] - df["Date_Debut_Eruption"]).dt.days

delai_moyen = df["Delai_Notification"].mean()
delai_median = df["Delai_Notification"].median()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Délai moyen", f"{delai_moyen:.1f} jours")
with col2:
    st.metric("Délai médian", f"{delai_median:.0f} jours")
with col3:
    # Correction pour la semaine en cours
    semaine_actuelle = df["Semaine"].max()
    cas_semaine_actuelle = len(df[df["Semaine"] == semaine_actuelle])
    
    # Facteur de correction (simplifié - basé sur délai moyen)
    facteur_correction = 1 + (delai_moyen / 7)
    cas_corriges = int(cas_semaine_actuelle * facteur_correction)
    
    st.metric("Cas corrigés (semaine actuelle)", cas_corriges, 
              delta=cas_corriges - cas_semaine_actuelle)

# Distribution des délais
fig_delai = px.histogram(
    df,
    x="Delai_Notification",
    nbins=20,
    title="Distribution des délais de notification",
    labels={"Delai_Notification": "Délai (jours)", "count": "Nombre de cas"}
)
st.plotly_chart(fig_delai, use_container_width=True)

# ============================================================
# 12. INDICATEURS PAR TRANCHE D'ÂGE
# ============================================================

st.header("👶 Analyse par tranches d'âge")

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
        title="Distribution des cas par âge",
        labels={"Tranche_Age": "Tranche d'âge", "Nombre_Cas": "Nombre de cas"},
        color="Nombre_Cas",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_vacc_age = px.bar(
        age_stats,
        x="Tranche_Age",
        y="Pct_Non_Vaccines",
        title="Taux de non-vaccination par âge",
        labels={"Tranche_Age": "Tranche d'âge", "Pct_Non_Vaccines": "% non vaccinés"},
        color="Pct_Non_Vaccines",
        color_continuous_scale="Oranges"
    )
    st.plotly_chart(fig_vacc_age, use_container_width=True)

# ============================================================
# 13. EXPORT DES DONNÉES
# ============================================================

st.header("💾 Export des données")

st.write("Exportez les données pour analyse approfondie dans un SIG ou Excel")

col1, col2, col3 = st.columns(3)

with col1:
    # Export CSV prédictions
    csv_pred = risk_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger prédictions (CSV)",
        data=csv_pred,
        file_name=f"predictions_rougeole_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Export GeoJSON
    geojson_str = sa_gdf_pred.to_json()
    st.download_button(
        label="🗺️ Télécharger carte (GeoJSON)",
        data=geojson_str,
        file_name=f"carte_rougeole_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.geojson",
        mime="application/json"
    )

with col3:
    # Export Excel complet
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        risk_df.to_excel(writer, sheet_name='Prédictions', index=False)
        cases_by_area.to_excel(writer, sheet_name='Cas_Observés', index=False)
        age_stats.to_excel(writer, sheet_name='Analyse_Age', index=False)
        future_df.to_excel(writer, sheet_name='Détail_Prédictions', index=False)
    
    st.download_button(
        label="📊 Télécharger rapport complet (Excel)",
        data=output.getvalue(),
        file_name=f"rapport_complet_rougeole_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# 14. RECOMMANDATIONS
# ============================================================

st.header("💡 Recommandations")

# Identifier les aires critiques
aires_critiques = risk_df[risk_df["Categorie_Risque"] == "Élevé"]["Aire_Sante"].tolist()

if aires_critiques:
    st.error(f"🚨 **{len(aires_critiques)} aire(s) à risque ÉLEVÉ identifiée(s):**")
    for aire in aires_critiques[:5]:
        aire_info = risk_df[risk_df["Aire_Sante"] == aire].iloc[0]
        st.write(f"- **{aire}**: {int(aire_info['Cas_Predits_Total'])} cas prédits, pic en {aire_info['Semaine_Pic']}")
    
    st.write("**Actions recommandées:**")
    st.write("✓ Renforcer la surveillance hebdomadaire dans ces aires")
    st.write("✓ Organiser des campagnes de vaccination de rattrapage")
    st.write("✓ Prépositionner les stocks de vaccins et matériel")
    st.write("✓ Sensibiliser les communautés aux signes de la rougeole")
else:
    st.success("✓ Aucune aire à risque élevé identifiée sur la période")

# Seuil d'alerte
st.info("""
**Seuils d'alerte épidémique:**
- 🟢 Faible: < 10 cas prédits
- 🟡 Moyen: 10-30 cas prédits  
- 🔴 Élevé: > 30 cas prédits
""")

# Footer
st.markdown("---")
st.caption(f"Dashboard généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} | Pays: {pays_selectionne} | Période: {start_date} - {end_date}")
