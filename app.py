"""
============================================================
APP COMPLET – SURVEILLANCE & PRÉDICTION ROUGEOLE (Multi-pays)
Version corrigée - Partie 1/3
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import ee
import geemap
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
from shapely.geometry import shape

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Surveillance Rougeole Multi-pays",
    layout="wide",
    page_icon="🦠",
)

# CSS personnalisé
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

# ============================================================
# 2. SIDEBAR – CONFIGURATION
# ============================================================
st.sidebar.header("📂 Configuration")

# Sélection pays
pays_selectionne = st.sidebar.selectbox(
    "Sélectionner le pays", 
    ["Niger", "Burkina Faso", "Mali", "Sénégal", "Tchad"]
)

# Aires de santé
st.sidebar.subheader("Aires de Santé")
option_aire = st.sidebar.radio(
    "Source des données géographiques", 
    ["GAUL Admin3 (GEE)", "Upload Shapefile/GeoJSON"]
)

upload_file = None
if option_aire == "Upload Shapefile/GeoJSON":
    upload_file = st.sidebar.file_uploader(
        "Charger un fichier géographique", 
        type=["shp", "geojson", "zip"]
    )

# Linelist
st.sidebar.subheader("Données épidémiologiques")
option_linelist = st.sidebar.radio(
    "Source des cas de rougeole",
    ["Upload CSV", "Données fictives (test)"]
)

linelist_file = None
if option_linelist == "Upload CSV":
    linelist_file = st.sidebar.file_uploader(
        "Linelists rougeole (CSV)", 
        type=["csv"],
        help="Format: ID_Cas, Date_Debut_Eruption, Date_Notification, Aire_Sante, Age_Mois, Statut_Vaccinal"
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
# 3. FONCTIONS DE CHARGEMENT AIRES DE SANTÉ
# ============================================================

@st.cache_data
def ee_fc_to_gdf_safe(_fc):
    """Convertir FeatureCollection GEE en GeoDataFrame"""
    try:
        # Extraction manuelle sans geemap
        features_info = _fc.limit(200).getInfo()
        
        if not features_info or 'features' not in features_info:
            raise ValueError("Aucune donnée retournée par GEE")
        
        features_list = features_info['features']
        
        geometries = []
        properties_list = []
        
        for feat in features_list:
            try:
                geom_dict = feat['geometry']
                props = feat['properties']
                
                # Utiliser shapely pour convertir
                geom = shape(geom_dict)
                geometries.append(geom)
                properties_list.append(props)
            except:
                continue
        
        if not geometries:
            raise ValueError("Aucune géométrie valide")
        
        gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs="EPSG:4326")
        
        # Standardisation des noms de colonnes
        if "ADM3_NAME" not in gdf.columns:
            if "ADM2_NAME" in gdf.columns:
                gdf["ADM3_NAME"] = gdf["ADM2_NAME"]
            elif "name" in gdf.columns:
                gdf["ADM3_NAME"] = gdf["name"]
            elif "NAME" in gdf.columns:
                gdf["ADM3_NAME"] = gdf["NAME"]
        
        return gdf
    except Exception as e:
        st.error(f"Erreur lors de la conversion GEE: {e}")
        return gpd.GeoDataFrame()

def load_shapefile_from_upload(upload_file):
    """Charge un shapefile depuis un upload"""
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
            for col in ["name", "NAME", "nom", "name_fr", "health_area", "ADMIN03"]:
                if col in gdf.columns:
                    gdf["ADM3_NAME"] = gdf[col]
                    break
            else:
                gdf["ADM3_NAME"] = [f"Aire_{i}" for i in range(len(gdf))]
        
        return gdf
        
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {e}")
        return gpd.GeoDataFrame()

# Chargement des aires de santé
with st.spinner("🔄 Chargement des aires de santé..."):
    if option_aire == "GAUL Admin3 (GEE)" and gee_ok:
        try:
            fc = ee.FeatureCollection("FAO/GAUL/2015/level2") \
                   .filter(ee.Filter.eq("ADM0_NAME", pays_selectionne))
            sa_gdf = ee_fc_to_gdf_safe(fc)
            
            if sa_gdf.empty:
                st.error("❌ Impossible de charger les données GAUL. Veuillez uploader un fichier.")
                st.stop()
            else:
                st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (GAUL)")
                
        except Exception as e:
            st.error(f"Erreur GAUL: {e}")
            st.stop()
            
    elif option_aire == "Upload Shapefile/GeoJSON":
        if upload_file is None:
            st.warning("⚠️ Veuillez uploader un fichier shapefile ou GeoJSON")
            st.stop()
        else:
            sa_gdf = load_shapefile_from_upload(upload_file)
            if sa_gdf.empty:
                st.error("❌ Fichier invalide ou vide")
                st.stop()
            else:
                st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (fichier)")
    else:
        st.error("❌ Google Earth Engine non disponible. Veuillez uploader un fichier.")
        st.stop()

# ============================================================
# 4. GÉNÉRATION/CHARGEMENT LINELIST
# ============================================================

@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500, start=None, end=None):
    """Génère des linelists réalistes pour tests"""
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
    
    return pd.DataFrame({
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

# Chargement des données de cas
if option_linelist == "Upload CSV" and linelist_file:
    try:
        df = pd.read_csv(linelist_file, parse_dates=["Date_Debut_Eruption", "Date_Notification"])
        st.sidebar.success(f"✓ {len(df)} cas chargés")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier CSV: {e}")
        st.stop()
else:
    df = generate_dummy_linelists(sa_gdf, start=start_date, end=end_date)
    st.sidebar.info("📊 Données fictives utilisées")

# Filtre temporel
df = df[
    (df["Date_Debut_Eruption"] >= pd.to_datetime(start_date)) & 
    (df["Date_Debut_Eruption"] <= pd.to_datetime(end_date))
]

if len(df) == 0:
    st.warning("⚠️ Aucun cas dans la période sélectionnée")
    st.stop()

# ============================================================
# 5. ENRICHISSEMENT DONNÉES SPATIALES - POPULATION
# ============================================================

@st.cache_data
def worldpop_children_stats(_sa_gdf, use_gee):
    """Extraction population WorldPop (0-14 ans)"""
    if not use_gee:
        return pd.DataFrame({
            "ADM3_NAME": _sa_gdf["ADM3_NAME"], 
            "Pop_Totale": [0] * len(_sa_gdf),
            "Pop_Moins_15": [0] * len(_sa_gdf)
        })
    
    try:
        # Convertir GeoDataFrame en FeatureCollection manuellement
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"ADM3_NAME": row["ADM3_NAME"]}
            
            # Convertir la géométrie en format GEE
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
        bands_age = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
        pop = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic()
        
        # Sélectionner les bandes enfants
        male_bands = [f"M{b}" for b in bands_age[:5]]
        female_bands = [f"F{b}" for b in bands_age[:5]]
        
        pop_children = pop.select(male_bands + female_bands)
        
        stats = pop_children.reduceRegions(
            collection=fc, 
            reducer=ee.Reducer.sum(), 
            scale=100
        )
        
        # Convertir manuellement
        stats_info = stats.getInfo()
        
        data_list = []
        for feat in stats_info['features']:
            props = feat['properties']
            pop_sum = sum([props.get(band, 0) for band in male_bands + female_bands])
            data_list.append({
                "ADM3_NAME": props.get("ADM3_NAME", ""),
                "Pop_Moins_15": pop_sum,
                "Pop_Totale": pop_sum * 2.2
            })
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.warning(f"⚠️ WorldPop indisponible: {e}")
        return pd.DataFrame({
            "ADM3_NAME": _sa_gdf["ADM3_NAME"], 
            "Pop_Totale": [0] * len(_sa_gdf),
            "Pop_Moins_15": [0] * len(_sa_gdf)
        })

with st.spinner("🔄 Extraction données population..."):
    pop_df = worldpop_children_stats(sa_gdf, gee_ok)

# ============================================================
# 6. URBANISATION – GHSL SMOD
# ============================================================

@st.cache_data
def urban_classification(_sa_gdf, use_gee):
    """Classification urbain/rural via GHSL"""
    if not use_gee:
        return pd.DataFrame({
            "ADM3_NAME": _sa_gdf["ADM3_NAME"],
            "Urbanisation": ["Rural"] * len(_sa_gdf)
        })
    
    try:
        # Convertir GeoDataFrame en FeatureCollection manuellement
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"ADM3_NAME": row["ADM3_NAME"]}
            
            # Convertir la géométrie en format GEE
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
            
            # Classification: 30=urbain, 23=semi-urbain, 21-22=rural, autres=rural
            urbanisation = ee.Algorithms.If(
                smod_value.gte(30), "Urbain",
                ee.Algorithms.If(smod_value.eq(23), "Semi-urbain", "Rural")
            )
            
            return feature.set({"Urbanisation": urbanisation})
        
        urban_fc = fc.map(classify)
        
        # Convertir manuellement
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
        st.sidebar.warning(f"⚠️ GHSL indisponible: {e}")
        return pd.DataFrame({
            "ADM3_NAME": _sa_gdf["ADM3_NAME"],
            "Urbanisation": ["Rural"] * len(_sa_gdf)
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
                
                # Moyennes
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
                    "Temperature_Moy": 28.0,
                    "Humidite_Moy": 50.0,
                    "Saison_Seche_Humidite": 35.0
                })
                
        except Exception as e:
            data_list.append({
                "ADM3_NAME": row["ADM3_NAME"],
                "Temperature_Moy": 28.0,
                "Humidite_Moy": 50.0,
                "Saison_Seche_Humidite": 35.0
            })
    
    return pd.DataFrame(data_list)

with st.spinner("🔄 Récupération données climatiques..."):
    climate_df = fetch_climate_nasa_power(sa_gdf, start_date, end_date)

# Fusionner toutes les données
sa_gdf_enrichi = sa_gdf.copy()
sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df, on="ADM3_NAME", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df, on="ADM3_NAME", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df, on="ADM3_NAME", how="left")

# Calculer superficie et densité
sa_gdf_enrichi["Superficie_km2"] = sa_gdf_enrichi.geometry.area / 1e6
sa_gdf_enrichi["Densite_Pop"] = sa_gdf_enrichi["Pop_Totale"] / sa_gdf_enrichi["Superficie_km2"]
sa_gdf_enrichi["Densite_Moins_15"] = sa_gdf_enrichi["Pop_Moins_15"] / sa_gdf_enrichi["Superficie_km2"]

# Remplacer NaN/Inf
sa_gdf_enrichi = sa_gdf_enrichi.replace([np.inf, -np.inf], 0)
sa_gdf_enrichi = sa_gdf_enrichi.fillna(0)
# ============================================================
# PARTIE 2/3 - ANALYSE ÉPIDÉMIOLOGIQUE ET CARTE
# ============================================================

# ============================================================
# 8. KPIs GLOBAUX
# ============================================================

st.header("📊 Indicateurs Clés")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Cas totaux", f"{len(df):,}")
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
# 10. CARTE INTERACTIVE - SITUATION ACTUELLE
# ============================================================

st.header("🗺️ Cartographie de la Situation Actuelle")

center_lat = sa_gdf_with_cases.geometry.centroid.y.mean()
center_lon = sa_gdf_with_cases.geometry.centroid.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

# Colormap
import branca.colormap as cm
max_cases = sa_gdf_with_cases["Cas_Observes"].max()

if max_cases > 0:
    colormap = cm.LinearColormap(
        colors=['#4caf50', '#ffeb3b', '#ff9800', '#f44336'],
        vmin=0,
        vmax=max_cases,
        caption="Nombre de cas observés"
    )
    colormap.add_to(m)

# Ajouter les polygones
for idx, row in sa_gdf_with_cases.iterrows():
    popup_html = f"""
    <div style="font-family: Arial; width: 300px;">
        <h4 style="margin-bottom: 10px; color: #1976d2;">{row['ADM3_NAME']}</h4>
        <hr style="margin: 5px 0;">
        <b>📊 Données observées:</b><br>
        • Cas observés: <b>{int(row['Cas_Observes'])}</b><br>
        • Non vaccinés: {row['Taux_Non_Vaccines']:.1f}%<br>
        • Âge moyen: {row.get('Age_Moyen', 0):.0f} mois<br>
        <hr style="margin: 5px 0;">
        <b>👥 Démographie:</b><br>
        • Population totale: {int(row['Pop_Totale']):,}<br>
        • Enfants &lt;15 ans: {int(row['Pop_Moins_15']):,}<br>
        • Densité: {row['Densite_Pop']:.1f} hab/km²<br>
        • Type: {row['Urbanisation']}<br>
        <hr style="margin: 5px 0;">
        <b>🌡️ Climat:</b><br>
        • Température: {row['Temperature_Moy']:.1f}°C<br>
        • Humidité: {row['Humidite_Moy']:.1f}%
    </div>
    """
    
    fill_color = colormap(row['Cas_Observes']) if max_cases > 0 else '#cccccc'
    
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, color=fill_color: {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.Tooltip(f"{row['ADM3_NAME']}: {int(row['Cas_Observes'])} cas"),
        popup=folium.Popup(popup_html, max_width=350)
    ).add_to(m)

# Heatmap
heat_data = [
    [row.geometry.centroid.y, row.geometry.centroid.x, row['Cas_Observes']] 
    for idx, row in sa_gdf_with_cases.iterrows() if row['Cas_Observes'] > 0
]

if heat_data:
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

st_folium(m, width=1400, height=600)

# ============================================================
# 11. ANALYSE TEMPORELLE
# ============================================================

st.header("📈 Évolution Temporelle")

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
fig_temporal.update_layout(hovermode="x unified")
st.plotly_chart(fig_temporal, use_container_width=True)

# ============================================================
# 12. ANALYSE PAR TRANCHE D'ÂGE
# ============================================================

st.subheader("👶 Distribution par âge")

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
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_vacc_age = px.bar(
        age_stats,
        x="Tranche_Age",
        y="Pct_Non_Vaccines",
        title="% non vaccinés par âge",
        color="Pct_Non_Vaccines",
        color_continuous_scale="Oranges"
    )
    st.plotly_chart(fig_vacc_age, use_container_width=True)

# ============================================================
# 13. NOWCASTING
# ============================================================

st.subheader("⏱️ Nowcasting - Délais de notification")

df["Delai_Notification"] = (df["Date_Notification"] - df["Date_Debut_Eruption"]).dt.days

delai_moyen = df["Delai_Notification"].mean()
delai_median = df["Delai_Notification"].median()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Délai moyen", f"{delai_moyen:.1f} jours")
with col2:
    st.metric("Délai médian", f"{delai_median:.0f} jours")
with col3:
    semaine_actuelle = df["Semaine"].max()
    cas_semaine_actuelle = len(df[df["Semaine"] == semaine_actuelle])
    facteur_correction = 1 + (delai_moyen / 7)
    cas_corriges = int(cas_semaine_actuelle * facteur_correction)
    st.metric("Cas corrigés (semaine actuelle)", cas_corriges, 
              delta=cas_corriges - cas_semaine_actuelle)
# ============================================================
# PARTIE 3/3 - MODÉLISATION PRÉDICTIVE
# ============================================================

st.header("🔮 Modélisation Prédictive")

st.info("""
**Note:** La prédiction utilise un modèle d'apprentissage automatique qui prend en compte:
- Historique des cas par aire de santé
- Données démographiques (population, densité)
- Urbanisation
- Données climatiques (température, humidité)
- Taux de vaccination
""")

# Bouton pour lancer la prédiction
lancer_prediction = st.button("🚀 Lancer la Prédiction", type="primary", use_container_width=True)

if lancer_prediction or st.session_state.get("prediction_lancee", False):
    st.session_state["prediction_lancee"] = True
    
    with st.spinner("🤖 Entraînement du modèle et génération des prédictions..."):
        
        # ============================================================
        # PRÉPARATION DES DONNÉES
        # ============================================================
        
        weekly_features = df.groupby(["Aire_Sante", "Semaine"]).agg(
            Cas_Observes=("ID_Cas", "count"),
            Non_Vaccines=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100),
            Age_Moyen=("Age_Mois", "mean")
        ).reset_index()
        
        # Fusionner avec données spatiales
        weekly_features = weekly_features.merge(
            sa_gdf_enrichi[["ADM3_NAME", "Pop_Totale", "Pop_Moins_15", "Densite_Moins_15", 
                             "Urbanisation", "Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite"]],
            left_on="Aire_Sante",
            right_on="ADM3_NAME",
            how="left"
        )
        
        # Encoder variables catégorielles
        le_urban = LabelEncoder()
        weekly_features["Urban_Encoded"] = le_urban.fit_transform(
            weekly_features["Urbanisation"].fillna("Rural")
        )
        
        # Features pour le modèle
        feature_cols = [
            "Non_Vaccines", "Age_Moyen",
            "Pop_Totale", "Pop_Moins_15", "Densite_Moins_15",
            "Urban_Encoded", "Temperature_Moy", "Saison_Seche_Humidite"
        ]
        
        # Remplacer NaN et Inf
        for col in feature_cols:
            weekly_features[col] = weekly_features[col].replace([np.inf, -np.inf], 0)
            weekly_features[col] = weekly_features[col].fillna(weekly_features[col].mean())
        
        X = weekly_features[feature_cols]
        y = weekly_features["Cas_Observes"]
        
        # Vérification des données
        if X.isnull().any().any() or np.isinf(X.values).any():
            st.error("❌ Données invalides détectées. Vérifiez vos données d'entrée.")
            st.stop()
        
        # ============================================================
        # ENTRAÎNEMENT DU MODÈLE
        # ============================================================
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=4, 
            random_state=42
        )
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        st.success(f"✓ Modèle entraîné avec succès - Score R²: {score:.3f}")
        
        # Importance des variables
        feature_importance = pd.DataFrame({
            "Variable": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        with st.expander("📊 Importance des facteurs prédictifs"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(
                    feature_importance.style.format({"Importance": "{:.3f}"}),
                    height=300
                )
            
            with col2:
                fig_importance = px.bar(
                    feature_importance,
                    x="Importance",
                    y="Variable",
                    orientation="h",
                    title="Importance des variables"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # ============================================================
        # GÉNÉRATION DES PRÉDICTIONS
        # ============================================================
        
        st.subheader(f"📅 Prédictions - {n_weeks_pred} semaines")
        
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
        
        # Vérifier les données futures
        for col in feature_cols:
            future_df[col] = future_df[col].replace([np.inf, -np.inf], 0)
            future_df[col] = future_df[col].fillna(0)
        
        future_df["Predicted_Cases"] = model.predict(future_df[feature_cols]).clip(0)
        
        # ============================================================
        # CLASSEMENT DES AIRES À RISQUE
        # ============================================================
        
        risk_df = future_df.groupby("Aire_Sante").agg(
            Cas_Predits_Total=("Predicted_Cases", "sum"),
            Cas_Predits_Max=("Predicted_Cases", "max"),
            Semaine_Pic=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(), "Semaine"] if len(x) > 0 else "N/A")
        ).reset_index()
        
        risk_df = risk_df.sort_values("Cas_Predits_Total", ascending=False)
        
        # Catégories de risque
        risk_df["Categorie_Risque"] = pd.cut(
            risk_df["Cas_Predits_Total"],
            bins=[0, 10, 30, np.inf],
            labels=["Faible", "Moyen", "Élevé"]
        )
        
        # ============================================================
        # VISUALISATIONS
        # ============================================================
        
        # Top N et Bottom N
        st.subheader("🔝 Top 10 Aires à Risque Élevé")
        
        top_10 = risk_df.head(10)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Graphique d'évolution
            future_top10 = future_df[future_df["Aire_Sante"].isin(top_10["Aire_Sante"])]
            
            fig_pred = px.line(
                future_top10,
                x="Semaine_Num",
                y="Predicted_Cases",
                color="Aire_Sante",
                title=f"Évolution prédite - Top 10 aires ({n_weeks_pred} semaines)",
                labels={"Semaine_Num": "Semaine", "Predicted_Cases": "Cas prédits"}
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            # Distribution des risques
            fig_risk_dist = px.pie(
                risk_df,
                names="Categorie_Risque",
                title="Distribution des risques",
                color="Categorie_Risque",
                color_discrete_map={"Faible": "#4caf50", "Moyen": "#ff9800", "Élevé": "#f44336"}
            )
            st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        # Tableau Top 10
        def highlight_risk(row):
            if row["Categorie_Risque"] == "Élevé":
                return ["background-color: #ffcdd2"] * len(row)
            elif row["Categorie_Risque"] == "Moyen":
                return ["background-color: #ffe0b2"] * len(row)
            else:
                return ["background-color: #c8e6c9"] * len(row)
        
        st.dataframe(
            top_10.style.apply(highlight_risk, axis=1).format({
                "Cas_Predits_Total": "{:.0f}",
                "Cas_Predits_Max": "{:.0f}"
            }),
            use_container_width=True
        )
        
        # Bottom 10 (amélioration)
        st.subheader("✅ Top 10 Aires avec Amélioration Attendue")
        
        bottom_10 = risk_df.tail(10).sort_values("Cas_Predits_Total")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_bottom = px.bar(
                bottom_10,
                x="Aire_Sante",
                y="Cas_Predits_Total",
                title="Aires avec faible risque prédit",
                color="Categorie_Risque",
                color_discrete_map={"Faible": "#4caf50", "Moyen": "#ff9800"}
            )
            fig_bottom.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bottom, use_container_width=True)
        
        with col2:
            st.dataframe(
                bottom_10[["Aire_Sante", "Cas_Predits_Total", "Categorie_Risque"]].style.format({
                    "Cas_Predits_Total": "{:.0f}"
                }),
                use_container_width=True
            )
        
        # ============================================================
        # CARTE PRÉDICTIVE
        # ============================================================
        
        st.subheader("🗺️ Carte des Prédictions")
        
        sa_gdf_pred = sa_gdf_enrichi.merge(
            risk_df,
            left_on="ADM3_NAME",
            right_on="Aire_Sante",
            how="left"
        )
        sa_gdf_pred["Cas_Predits_Total"] = sa_gdf_pred["Cas_Predits_Total"].fillna(0)
        
        m_pred = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=6, 
            tiles="CartoDB positron"
        )
        
        max_pred = sa_gdf_pred["Cas_Predits_Total"].max()
        
        if max_pred > 0:
            colormap_pred = cm.LinearColormap(
                colors=['#4caf50', '#ffeb3b', '#ff9800', '#f44336'],
                vmin=0,
                vmax=max_pred,
                caption="Cas prédits (total)"
            )
            colormap_pred.add_to(m_pred)
        
        for idx, row in sa_gdf_pred.iterrows():
            popup_html = f"""
            <div style="font-family: Arial; width: 320px;">
                <h4 style="color: #1976d2;">{row['ADM3_NAME']}</h4>
                <hr>
                <b>🔮 Prédictions ({n_weeks_pred} sem):</b><br>
                • Total prédit: <b style="font-size: 16px;">{int(row['Cas_Predits_Total'])}</b><br>
                • Pic attendu: {int(row['Cas_Predits_Max'])} cas<br>
                • Semaine du pic: {row['Semaine_Pic']}<br>
                • Niveau de risque: <span style="color: {'#f44336' if row.get('Categorie_Risque')=='Élevé' else '#ff9800' if row.get('Categorie_Risque')=='Moyen' else '#4caf50'}; font-weight: bold;">{row.get('Categorie_Risque', 'N/A')}</span>
                <hr>
                <b>📊 Données observées:</b><br>
                • Cas actuels: {int(row.get('Cas_Observes', 0))}<br>
                • Population: {int(row['Pop_Totale']):,} hab.
            </div>
            """
            
            fill_color = colormap_pred(row['Cas_Predits_Total']) if max_pred > 0 else '#cccccc'
            
            folium.GeoJson(
                row['geometry'],
                style_function=lambda x, color=fill_color: {
                    'fillColor': color,
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                tooltip=folium.Tooltip(f"{row['ADM3_NAME']}: {int(row['Cas_Predits_Total'])} cas prédits"),
                popup=folium.Popup(popup_html, max_width=350)
            ).add_to(m_pred)
        
        st_folium(m_pred, width=1400, height=600)
        
        # ============================================================
        # HEATMAP TEMPORELLE
        # ============================================================
        
        st.subheader("📊 Heatmap Temporelle")
        
        heatmap_data = future_df.pivot(
            index="Aire_Sante",
            columns="Semaine",
            values="Predicted_Cases"
        ).fillna(0)
        
        top_15_aires = risk_df.head(15)["Aire_Sante"].tolist()
        heatmap_data_top = heatmap_data.loc[heatmap_data.index.isin(top_15_aires)]
        
        fig_heatmap = px.imshow(
            heatmap_data_top,
            labels=dict(x="Semaine", y="Aire de Santé", color="Cas prédits"),
            x=heatmap_data_top.columns,
            y=heatmap_data_top.index,
            title=f"Évolution prédite - Top 15 aires ({n_weeks_pred} semaines)",
            color_continuous_scale="Reds",
            aspect="auto"
        )
        fig_heatmap.update_xaxes(side="bottom")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # ============================================================
        # RECOMMANDATIONS
        # ============================================================
        
        st.header("💡 Recommandations")
        
        aires_critiques = risk_df[risk_df["Categorie_Risque"] == "Élevé"]["Aire_Sante"].tolist()
        
        if aires_critiques:
            st.error(f"🚨 **{len(aires_critiques)} aire(s) à risque ÉLEVÉ identifiée(s)**")
            
            for i, aire in enumerate(aires_critiques[:5], 1):
                aire_info = risk_df[risk_df["Aire_Sante"] == aire].iloc[0]
                st.write(f"{i}. **{aire}**: {int(aire_info['Cas_Predits_Total'])} cas prédits, pic en {aire_info['Semaine_Pic']}")
            
            st.write("**Actions prioritaires recommandées:**")
            st.write("✓ Renforcer la surveillance épidémiologique hebdomadaire")
            st.write("✓ Organiser des campagnes de vaccination de rattrapage")
            st.write("✓ Prépositionner stocks de vaccins et intrants")
            st.write("✓ Sensibiliser les communautés aux signes d'alerte")
            st.write("✓ Coordonner avec les partenaires (OMS, UNICEF)")
        else:
            st.success("✓ Aucune aire à risque élevé identifiée")
        
        # ============================================================
        # EXPORT DES RÉSULTATS
        # ============================================================
        
        st.header("💾 Export des Résultats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_pred = risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Prédictions (CSV)",
                data=csv_pred,
                file_name=f"predictions_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            geojson_str = sa_gdf_pred.to_json()
            st.download_button(
                label="🗺️ Carte (GeoJSON)",
                data=geojson_str,
                file_name=f"carte_predictions_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.geojson",
                mime="application/json"
            )
        
        with col3:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                risk_df.to_excel(writer, sheet_name='Prédictions', index=False)
                cases_by_area.to_excel(writer, sheet_name='Cas_Observés', index=False)
                age_stats.to_excel(writer, sheet_name='Analyse_Age', index=False)
                future_df.to_excel(writer, sheet_name='Détail_Semaines', index=False)
            
            st.download_button(
                label="📊 Rapport complet (Excel)",
                data=output.getvalue(),
                file_name=f"rapport_{pays_selectionne}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("👆 Cliquez sur le bouton ci-dessus pour lancer la prédiction")

# Footer
st.markdown("---")
st.caption(f"Dashboard généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} | Pays: {pays_selectionne} | Période: {start_date} - {end_date}")
