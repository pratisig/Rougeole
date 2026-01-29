import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import ee
import geemap  # On importe geemap standard pour les conversions de données
import folium
from streamlit_folium import st_folium
import plotly.express as px
import json

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Surveillance Rougeole - Ouest Afrique", layout="wide", page_icon="🦠")
st.title("🦠 Dashboard de Surveillance Prédictive – Rougeole")

# ============================================================
# 1. INITIALISATION GEE (Compatible Cloud & Local)
# ============================================================
@st.cache_resource
def init_gee():
    try:
        # TENTATIVE 1 : AUTH VIA SECRETS (POUR LE CLOUD)
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"],
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
        # TENTATIVE 2 : AUTH LOCALE (Via gcloud)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"⚠️ Erreur d’authentification GEE: {e}")
        st.info("Astuce : Pour le local, lancez 'earthengine authenticate' dans votre terminal.")
        return False

if not init_gee():
    st.stop()

# ============================================================
# 2. SIDEBAR - PARAMÈTRES
# ============================================================
st.sidebar.header("📂 Configuration")
pays_selectionne = st.sidebar.selectbox("Pays", ["Niger", "Burkina Faso", "Mali"])
option_aire = st.sidebar.radio("Source Géographique", ["GAUL Admin3 (GEE)", "Upload GeoJSON"])
linelist_file = st.sidebar.file_uploader("Linelists (CSV)", type=["csv"], help="Colonnes requises: Aire_Sante, Date_Debut_Eruption, Statut_Vaccinal")

# ============================================================
# 3. GESTION DES DONNÉES GÉOGRAPHIQUES
# ============================================================
@st.cache_resource
def get_boundaries(option, pays, uploaded_file):
    """Récupère les géométries et les convertit en GeoDataFrame utilisable"""
    try:
        if option == "GAUL Admin3 (GEE)":
            # Chargement depuis les serveurs Google
            fc = ee.FeatureCollection("FAO/GAUL/2015/level3").filter(ee.Filter.eq("ADM0_NAME", pays))
            # Conversion GEE -> GeoDataFrame (Peut être long, on met un spinner ailleurs)
            gdf = geemap.ee_to_gdf(fc)
            # Standardisation
            if "ADM3_NAME" in gdf.columns:
                gdf = gdf.rename(columns={"ADM3_NAME": "Aire_Sante"})
            return fc, gdf
        
        elif option == "Upload GeoJSON" and uploaded_file:
            gdf = gpd.read_file(uploaded_file)
            # Détection colonne nom
            col_name = next((c for c in gdf.columns if c.lower() in ["aire_sante", "health_area", "adm3_name", "nom"]), None)
            if col_name:
                gdf = gdf.rename(columns={col_name: "Aire_Sante"})
            else:
                st.error("Le fichier doit contenir une colonne nommant l'aire de santé.")
                return None, None
            # Conversion GeoDataFrame -> GEE
            fc = geemap.gdf_to_ee(gdf)
            return fc, gdf
    except Exception as e:
        st.error(f"Erreur lors du chargement des frontières : {e}")
        return None, None
    return None, None

with st.spinner("Chargement des frontières (GEE/Fichier)..."):
    ee_fc, gdf_boundaries = get_boundaries(option_aire, pays_selectionne, linelist_file if option_aire == "Upload GeoJSON" else None)

if gdf_boundaries is None:
    st.info("👈 Veuillez configurer les sources de données dans le menu.")
    st.stop()

# ============================================================
# 4. DONNÉES ÉPIDÉMIOLOGIQUES (LINELIST)
# ============================================================
@st.cache_data
def load_linelist(file, aires_ref):
    if file:
        df = pd.read_csv(file, parse_dates=["Date_Debut_Eruption"])
        # Vérification colonnes
        req_cols = ["Date_Debut_Eruption", "Aire_Sante"]
        if not all(c in df.columns for c in req_cols):
            st.error(f"Le CSV doit contenir : {req_cols}")
            return None
    else:
        # --- MODE DÉMO : GÉNÉRATION DONNÉES FICTIVES MAIS COHÉRENTES ---
        n = 500
        dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 180, n), unit="D")
        real_areas = aires_ref
        df = pd.DataFrame({
            "ID_Cas": range(n),
            "Date_Debut_Eruption": dates,
            "Aire_Sante": np.random.choice(real_areas, n),
            "Statut_Vaccinal": np.random.choice(["Oui", "Non"], n, p=[0.4, 0.6])
        })
    
    df["Semaine"] = df["Date_Debut_Eruption"].dt.isocalendar().week
    return df

df_linelist = load_linelist(linelist_file, gdf_boundaries["Aire_Sante"].unique())
if df_linelist is None: st.stop()

# ============================================================
# 5. ENRICHISSEMENT SATELLITAIRE (Population & Urbanisation)
# ============================================================
# On utilise geemap pour extraire les stats zonales, mais sans afficher la map geemap
@st.cache_data
def extract_satellite_features(_feature_collection):
    # Note: _feature_collection avec underscore empêche streamlit de hasher l'objet EE (lent)
    try:
        # A. POPULATION (WorldPop)
        pop_img = ee.ImageCollection("WorldPop/GPW/v11/population").first()
        # Somme de la pop par polygone
        pop_stats = pop_img.reduceRegions(collection=_feature_collection, reducer=ee.Reducer.sum(), scale=1000)
        pop_df = geemap.ee_to_pandas(pop_stats)
        
        # B. URBANISATION (JRC GHSL)
        urban_img = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")
        urban_stats = urban_img.reduceRegions(collection=_feature_collection, reducer=ee.Reducer.mode(), scale=1000)
        urban_df = geemap.ee_to_pandas(urban_stats)
        
        # NETTOYAGE ET FUSION PANDAS
        # Identifier la colonne de jointure (souvent ADM3_NAME ou Aire_Sante selon la source)
        col_join = "Aire_Sante" if "Aire_Sante" in pop_df.columns else "ADM3_NAME"
        
        pop_df = pop_df.rename(columns={col_join: "Aire_Sante", "population": "Pop_Totale"})[[ "Aire_Sante", "Pop_Totale"]]
        urban_df = urban_df.rename(columns={col_join: "Aire_Sante", "mode": "Code_Urbanisation"})[[ "Aire_Sante", "Code_Urbanisation"]]
        
        final_sat_df = pop_df.merge(urban_df, on="Aire_Sante")
        return final_sat_df
        
    except Exception as e:
        st.error(f"Erreur extraction GEE: {e}")
        return pd.DataFrame() # Retourne vide en cas d'erreur

with st.spinner("Analyse des images satellitaires (GEE)..."):
    df_sat = extract_satellite_features(ee_fc)

if df_sat.empty:
    st.warning("Impossible de récupérer les données satellites. Le modèle utilisera des valeurs par défaut.")
    # Fallback pour ne pas planter l'app
    df_sat = pd.DataFrame({"Aire_Sante": gdf_boundaries["Aire_Sante"].unique(), "Pop_Totale": 10000, "Code_Urbanisation": 10})

# ============================================================
# 6. API MÉTÉO (NASA POWER)
# ============================================================
# Pour la prod, on peut appeler l'API. Ici, simulation pour rapidité
def get_climate_data(df):
    # Simule l'humidité relative (RH) - Facteur clé rougeole
    # En prod: Boucler sur lat/lon des centroids et appeler API NASA
    df["Humidite_Relative"] = np.random.uniform(10, 80, len(df)) # 10-30% = Risque élevé (Saison sèche)
    return df

# ============================================================
# 7. MACHINE LEARNING & PRÉDICTION
# ============================================================
st.subheader("🔮 Prédiction & Analyse de Risque")

# 1. Préparation du Dataset d'entrainement
weekly_stats = df_linelist.groupby(["Aire_Sante", "Semaine"]).agg(
    Cas_Observes=("ID_Cas", "count"),
    Non_Vaccines_Pct=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100)
).reset_index()

# Fusion Géographie + Satellites
ml_df = weekly_stats.merge(df_sat, on="Aire_Sante", how="left").fillna(0)
ml_df = get_climate_data(ml_df)

# Encodage Urbanisation
le = LabelEncoder()
ml_df["Urban_Encoded"] = le.fit_transform(ml_df["Code_Urbanisation"].astype(str))

# Entrainement
features = ["Semaine", "Non_Vaccines_Pct", "Pop_Totale", "Urban_Encoded", "Humidite_Relative"]
X = ml_df[features]
y = ml_df["Cas_Observes"]

model = GradientBoostingRegressor(n_estimators=100)
model.fit(X, y)

# Prédiction S+4 (4 prochaines semaines)
future_rows = []
last_week = ml_df["Semaine"].max()

for aire in ml_df["Aire_Sante"].unique():
    base = ml_df[ml_df["Aire_Sante"] == aire].iloc[-1]
    for i in range(1, 5):
        row = base.copy()
        row["Semaine"] = last_week + i
        # Hypothèse: l'humidité baisse (saison sèche approche)
        row["Humidite_Relative"] = max(10, row["Humidite_Relative"] - 5) 
        future_rows.append(row)

df_future = pd.DataFrame(future_rows)
df_future["Cas_Prevus"] = model.predict(df_future[features]).clip(lower=0).round().astype(int)

# ============================================================
# 8. VISUALISATION (FOLIUM PUR)
# ============================================================
# On utilise Folium directement pour éviter le bug geemap.foliumap

col_map, col_stats = st.columns([2, 1])

with col_map:
    st.write("#### 🗺️ Carte de Risque (Prévision S+4)")
    
    # Agrégation des prédictions par aire
    map_pred = df_future.groupby("Aire_Sante")["Cas_Prevus"].sum().reset_index()
    
    # Jointure avec le GeoDataFrame
    gdf_final = gdf_boundaries.merge(map_pred, on="Aire_Sante", how="left").fillna(0)
    
    # Création de la carte
    # Centrage automatique
    centroid = gdf_final.geometry.centroid
    avg_lat, avg_lon = centroid.y.mean(), centroid.x.mean()
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6, tiles="CartoDB positron")
    
    # Choropleth
    folium.Choropleth(
        geo_data=gdf_final,
        name="Prévision Rougeole",
        data=gdf_final,
        columns=["Aire_Sante", "Cas_Prevus"],
        key_on="feature.properties.Aire_Sante",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Cas Prévus (Prochain mois)",
        highlight=True
    ).add_to(m)
    
    # Tooltip interactif
    folium.GeoJson(
        gdf_final,
        style_function=lambda x: {'color': 'transparent', 'fillColor': 'transparent'},
        tooltip=folium.GeoJsonTooltip(
            fields=['Aire_Sante', 'Cas_Prevus'],
            aliases=['Aire de Santé', 'Cas Prévus'],
            localize=True
        )
    ).add_to(m)
    
    st_folium(m, width=None, height=500)

with col_stats:
    st.write("#### 🚨 Top Alertes")
    top_risk = map_pred.sort_values("Cas_Prevus", ascending=False).head(10)
    st.dataframe(top_risk, hide_index=True)
    
    st.write("#### 📈 Tendance")
    # Graphique Plotly
    fig = px.line(df_future, x="Semaine", y="Cas_Prevus", color="Aire_Sante", title="Courbes épidémiques projetées")
    fig.update_layout(showlegend=False) # Trop chargé sinon
    st.plotly_chart(fig, use_container_width=True)

st.success("Application chargée avec succès.")
