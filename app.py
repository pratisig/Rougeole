import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import ee
# Correction de l'import pour éviter certains conflits
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import plotly.express as px
import json
import folium

st.set_page_config(page_title="Surveillance Rougeole Multi-pays", layout="wide", page_icon="🦠")
st.title("🦠 Dashboard de Surveillance Prédictive – Rougeole")

# ============================================================
# 1. INITIALISATION GEE (Compatible Cloud & Local)
# ============================================================
@st.cache_resource
def init_gee():
    try:
        # TENTATIVE D'AUTH VIA SECRETS (POUR LE CLOUD/PROD)
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"],
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
        else:
            # AUTH LOCALE (Via gcloud auth login)
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"Erreur d’authentification GEE: {e}")
        st.warning("Assurez-vous d'avoir configuré les secrets ou l'authentification locale.")
        return False

if not init_gee():
    st.stop()

# ============================================================
# 2. SIDEBAR
# ============================================================
st.sidebar.header("📂 Configuration")
pays_selectionne = st.sidebar.selectbox("Pays", ["Niger", "Burkina Faso", "Mali"])
option_aire = st.sidebar.radio("Source Géographique", ["GAUL Admin3 (GEE)", "Upload GeoJSON"])
linelist_file = st.sidebar.file_uploader("Linelists (CSV)", type=["csv"])

# ============================================================
# 3. CHARGEMENT AIRES DE SANTÉ & CONVERSION
# ============================================================
@st.cache_resource
def get_boundaries(option, pays, uploaded_file):
    if option == "GAUL Admin3 (GEE)":
        # On charge depuis GEE
        fc = ee.FeatureCollection("FAO/GAUL/2015/level3").filter(ee.Filter.eq("ADM0_NAME", pays))
        # Conversion en GeoDataFrame pour l'affichage Folium et le ML
        # geemap.ee_to_gdf convertit les données du serveur vers votre RAM Python
        gdf = geemap.ee_to_gdf(fc)
        # Standardisation des colonnes
        if "ADM3_NAME" in gdf.columns:
            gdf = gdf.rename(columns={"ADM3_NAME": "Aire_Sante"})
        return fc, gdf
    
    elif option == "Upload GeoJSON" and uploaded_file:
        gdf = gpd.read_file(uploaded_file)
        # Vérification colonne
        col_name = next((c for c in gdf.columns if "name" in c.lower() or "aire" in c.lower() or "health" in c.lower()), None)
        if col_name:
            gdf = gdf.rename(columns={col_name: "Aire_Sante"})
        else:
            st.error("Le fichier doit contenir une colonne nommant l'aire de santé.")
            st.stop()
            
        fc = geemap.gdf_to_ee(gdf)
        return fc, gdf
    return None, None

with st.spinner("Chargement et conversion des frontières (GEE vers Python)..."):
    ee_fc, gdf_boundaries = get_boundaries(option_aire, pays_selectionne, None if option_aire == "GAUL Admin3 (GEE)" else linelist_file)

if gdf_boundaries is None:
    st.info("En attente de sélection...")
    st.stop()

# ============================================================
# 4. DONNÉES LINELIST (SIMULÉES OU RÉELLES)
# ============================================================
@st.cache_data
def load_data(file, aires_disponibles):
    if file:
        df = pd.read_csv(file, parse_dates=["Date_Debut_Eruption", "Date_Notification"])
        # Standardisation noms colonnes si besoin
    else:
        # Simulation cohérente avec les polygones chargés
        n = 500
        dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 180, n), unit="D")
        
        # On choisit aléatoirement des aires qui existent vraiment dans le GeoJSON/GEE
        real_areas = aires_disponibles
        
        df = pd.DataFrame({
            "ID_Cas": range(n),
            "Date_Debut_Eruption": dates,
            "Date_Notification": dates + pd.to_timedelta(np.random.randint(1, 14, n), unit="D"),
            "Aire_Sante": np.random.choice(real_areas, n),
            "Age_Mois": np.random.randint(6, 180, n),
            "Statut_Vaccinal": np.random.choice(["Oui", "Non"], n, p=[0.4, 0.6])
        })
    
    df["Semaine"] = df["Date_Debut_Eruption"].dt.isocalendar().week
    return df

df_linelist = load_data(linelist_file, gdf_boundaries["Aire_Sante"].unique())

# ============================================================
# 5. EXTRACTION DONNÉES SATELLITAIRES (GEE -> PANDAS)
# ============================================================
# C'est l'étape manquante dans votre code original !
# Il faut extraire les pixels GEE pour en faire des chiffres dans le tableau Pandas

with st.spinner("Extraction des données Satellitaires (Population & Urbanisation)..."):
    try:
        # A. POPULATION (WorldPop)
        # On prend une image récente de densité de pop
        pop_img = ee.ImageCollection("WorldPop/GPW/v11/population").first()
        # Calcul de la somme de pop par polygone
        pop_stats = pop_img.reduceRegions(collection=ee_fc, reducer=ee.Reducer.sum(), scale=1000)
        # Conversion GEE -> Pandas
        pop_df = geemap.ee_to_pandas(pop_stats)
        
        # Gestion des noms de colonnes qui varient selon la source
        col_join_pop = "ADM3_NAME" if "ADM3_NAME" in pop_df.columns else "Aire_Sante"
        pop_df = pop_df.rename(columns={col_join_pop: "Aire_Sante", "population": "Pop_Totale"})
        # Nettoyage
        pop_df = pop_df[["Aire_Sante", "Pop_Totale"]].groupby("Aire_Sante").sum().reset_index()

        # B. URBANISATION (JRC GHSL)
        urban_img = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")
        urban_stats = urban_img.reduceRegions(collection=ee_fc, reducer=ee.Reducer.mode(), scale=1000)
        urban_df = geemap.ee_to_pandas(urban_stats)
        
        col_join_urb = "ADM3_NAME" if "ADM3_NAME" in urban_df.columns else "Aire_Sante"
        urban_df = urban_df.rename(columns={col_join_urb: "Aire_Sante", "mode": "Code_Urbanisation"})
        urban_df = urban_df[["Aire_Sante", "Code_Urbanisation"]]
        
    except Exception as e:
        st.error(f"Erreur extraction GEE: {e}")
        st.stop()

# ============================================================
# 6. PRÉPARATION ML & PRÉDICTION
# ============================================================
st.subheader("🔮 Modélisation Prédictive")

# Agrégation hebdomadaire (Combien de cas par aire par semaine ?)
weekly_df = df_linelist.groupby(["Aire_Sante", "Semaine"]).agg(
    Cas_Observes=("ID_Cas", "count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100)
).reset_index()

# Fusion de TOUTES les données (Geo + Linelist + Satellites)
ml_df = weekly_df.merge(pop_df, on="Aire_Sante", how="left")
ml_df = ml_df.merge(urban_df, on="Aire_Sante", how="left")

# Simulation données climatiques (Pour la démo rapide, car l'API NASA prend du temps pour 100+ points)
ml_df["Coef_Climatique"] = np.random.uniform(0.5, 1.5, len(ml_df)) 
ml_df = ml_df.fillna(0)

# Encodage du code urbanisation (classe 10, 11, 20, 30...)
le = LabelEncoder()
ml_df["Urban_Encoded"] = le.fit_transform(ml_df["Code_Urbanisation"].astype(str))

# Entrainement du modèle
features = ["Semaine", "Non_Vaccines", "Pop_Totale", "Urban_Encoded", "Coef_Climatique"]
X = ml_df[features]
y = ml_df["Cas_Observes"]

model = GradientBoostingRegressor(n_estimators=100)
model.fit(X, y)

# --- PRÉDICTION S+1 à S+4 ---
future_data = []
last_week = ml_df["Semaine"].max()

# Pour chaque aire de santé, on crée 4 lignes futures
for aire in ml_df["Aire_Sante"].unique():
    # On prend les caractéristiques fixes de l'aire (pop, urbanisation)
    base_info = ml_df[ml_df["Aire_Sante"] == aire].iloc[-1]
    
    for w in range(1, 5):
        row = base_info.copy()
        row["Semaine"] = last_week + w
        # Ici on pourrait injecter la météo prévue par la NASA
        future_data.append(row)

future_df = pd.DataFrame(future_data)
future_df["Cas_Prevus"] = model.predict(future_df[features])
# On arrondit car on ne peut pas avoir 0.5 cas
future_df["Cas_Prevus"] = future_df["Cas_Prevus"].clip(lower=0).round().astype(int)

# ============================================================
# 7. VISUALISATION FINALE
# ============================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Carte de Risque (Prévision S+4)")
    
    # On fait la somme des cas prévus sur 4 semaines par aire
    risk_map_data = future_df.groupby("Aire_Sante")["Cas_Prevus"].sum().reset_index()
    
    # Jointure avec le GeoDataFrame (Géométrie)
    map_final = gdf_boundaries.merge(risk_map_data, on="Aire_Sante", how="left").fillna(0)
    
    # Carte Folium
    m = folium.Map(location=[17, 9], zoom_start=6)
    
    folium.Choropleth(
        geo_data=map_final,
        name="Prédiction Rougeole",
        data=map_final,
        columns=["Aire_Sante", "Cas_Prevus"],
        key_on="feature.properties.Aire_Sante",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Total Cas Prévus (4 prochaines semaines)"
    ).add_to(m)
    
    # Affichage de la carte
    st_folium(m, width=700, height=500)

with col2:
    st.write("### 🚨 Aires en Alerte")
    # Top 10 des zones à risque
    top_risk = risk_map_data.sort_values("Cas_Prevus", ascending=False).head(10)
    st.dataframe(top_risk)

    st.write("### Tendance Temporelle")
    fig = px.line(future_df, x="Semaine", y="Cas_Prevus", color="Aire_Sante", 
                  title="Projection", markers=True)
    # On cache la légende si trop d'aires pour garder propre
    fig.update_layout(showlegend=False) 
    st.plotly_chart(fig, use_container_width=True)
