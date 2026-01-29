import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import plotly.express as px
import json
import folium
import branca.colormap as cm

st.set_page_config(page_title="Surveillance Rougeole Multi-pays", layout="wide", page_icon="🦠")
st.title("🦠 Dashboard de Surveillance Prédictive – Rougeole")

# ============================================================
# 1. INITIALISATION GEE
# ============================================================
@st.cache_resource
def init_gee():
    try:
        # TENTATIVE D'AUTH VIA SECRETS (POUR LE CLOUD)
        if "GEE_SERVICE_ACCOUNT" in st.secrets:
            key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"],
                key_data=json.dumps(key_dict)
            )
            ee.Initialize(credentials)
        else:
            # AUTH LOCALE (SI PAS DE SECRETS)
            ee.Initialize()
        return True
    except Exception as e:
        st.error(f"Erreur d’authentification GEE: {e}")
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
        # Conversion en GeoDataFrame pour l'affichage Folium et le mapping ML plus tard
        # Note: geemap.ee_to_gdf peut être lent sur de gros datasets, on limite ici
        gdf = geemap.ee_to_gdf(fc)
        # On renomme pour standardiser
        gdf = gdf.rename(columns={"ADM3_NAME": "Aire_Sante"})
        return fc, gdf
    
    elif option == "Upload GeoJSON" and uploaded_file:
        gdf = gpd.read_file(uploaded_file)
        if "Aire_Sante" not in gdf.columns:
            st.warning("Le fichier doit contenir une colonne 'Aire_Sante'")
            return None, None
        fc = geemap.gdf_to_ee(gdf)
        return fc, gdf
    return None, None

with st.spinner("Chargement des frontières..."):
    ee_fc, gdf_boundaries = get_boundaries(option_aire, pays_selectionne, None if option_aire == "GAUL Admin3 (GEE)" else linelist_file)

if gdf_boundaries is None:
    st.info("En attente de fichier ou de sélection...")
    st.stop()

# ============================================================
# 4. DONNÉES LINELIST (SIMULÉES OU RÉELLES)
# ============================================================
@st.cache_data
def load_data(file, aires_disponibles):
    if file:
        df = pd.read_csv(file, parse_dates=["Date_Debut_Eruption", "Date_Notification"])
    else:
        # Génération cohérente avec les aires chargées
        n = 500
        dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0, 180, n), unit="D")
        df = pd.DataFrame({
            "ID_Cas": range(n),
            "Date_Debut_Eruption": dates,
            "Date_Notification": dates + pd.to_timedelta(np.random.randint(1, 14, n), unit="D"),
            "Aire_Sante": np.random.choice(aires_disponibles, n),
            "Age_Mois": np.random.randint(6, 180, n),
            "Statut_Vaccinal": np.random.choice(["Oui", "Non"], n, p=[0.4, 0.6])
        })
    
    df["Semaine"] = df["Date_Debut_Eruption"].dt.isocalendar().week
    return df

df_linelist = load_data(linelist_file, gdf_boundaries["Aire_Sante"].unique())

# ============================================================
# 5. EXTRACTION DONNÉES SATELLITAIRES (GEE -> PANDAS)
# ============================================================
# C'est ici qu'il manquait la conversion vers Pandas dans votre code original

with st.spinner("Analyse satéllitaire (GEE)..."):
    # A. POPULATION
    pop_img = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex").mosaic().select(["M0","F0","M1","F1"])
    pop_stats = pop_img.reduceRegions(collection=ee_fc, reducer=ee.Reducer.sum(), scale=1000)
    # Conversion en DF
    pop_df = geemap.ee_to_pandas(pop_stats)
    # Calcul total enfants (somme des colonnes renvoyées)
    cols_pop = [c for c in pop_df.columns if c in ["M0","F0","M1","F1"]]
    pop_df["Pop_Totale"] = pop_df[cols_pop].sum(axis=1)
    # Standardisation nom colonne jointure (selon la source GEE, c'est souvent ADM3_NAME)
    col_name_join = "ADM3_NAME" if "ADM3_NAME" in pop_df.columns else "Aire_Sante"
    pop_df = pop_df.rename(columns={col_name_join: "Aire_Sante"})[["Aire_Sante", "Pop_Totale"]]

    # B. URBANISATION
    urban_img = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020")
    urban_stats = urban_img.reduceRegions(collection=ee_fc, reducer=ee.Reducer.mode(), scale=1000)
    urban_df = geemap.ee_to_pandas(urban_stats)
    urban_df = urban_df.rename(columns={col_name_join: "Aire_Sante", "mode": "Urbanisation"})[["Aire_Sante", "Urbanisation"]]

# ============================================================
# 6. PRÉPARATION ML & PRÉDICTION
# ============================================================
st.subheader("🔮 Modélisation Prédictive")

# Agrégation hebdomadaire
weekly_df = df_linelist.groupby(["Aire_Sante", "Semaine"]).agg(
    Cas_Observes=("ID_Cas", "count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100)
).reset_index()

# Jointures des données contextuelles
ml_df = weekly_df.merge(pop_df, on="Aire_Sante", how="left")
ml_df = ml_df.merge(urban_df, on="Aire_Sante", how="left")

# Simulation données climatiques (Pour l'exemple, car l'API NASA est lente en boucle)
# Dans la prod, faites l'appel API ici
ml_df["Coef_Climatique"] = np.random.uniform(0.5, 1.5, len(ml_df)) 

# Nettoyage
ml_df = ml_df.fillna(0)

# Encodage
le = LabelEncoder()
ml_df["Urban_Encoded"] = le.fit_transform(ml_df["Urbanisation"].astype(str))

# Entrainement
features = ["Semaine", "Non_Vaccines", "Pop_Totale", "Urban_Encoded", "Coef_Climatique"]
X = ml_df[features]
y = ml_df["Cas_Observes"]

model = GradientBoostingRegressor(n_estimators=100)
model.fit(X, y)

# Prédiction pour S+1 à S+4
future_data = []
last_week = ml_df["Semaine"].max()
for aire in ml_df["Aire_Sante"].unique():
    base_data = ml_df[ml_df["Aire_Sante"] == aire].iloc[-1]
    for w in range(1, 5):
        row = base_data.copy()
        row["Semaine"] = last_week + w
        future_data.append(row)

future_df = pd.DataFrame(future_data)
future_df["Cas_Prevus"] = model.predict(future_df[features])

# ============================================================
# 7. VISUALISATION FINALE
# ============================================================

col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Carte des Risques (Prévision S+4)")
    
    # Agrégation des prédictions par aire pour la carte
    risk_map_data = future_df.groupby("Aire_Sante")["Cas_Prevus"].sum().reset_index()
    
    # Jointure avec le GeoDataFrame des frontières
    map_final = gdf_boundaries.merge(risk_map_data, on="Aire_Sante", how="left").fillna(0)
    
    # Création carte Folium
    m = folium.Map(location=[17, 9], zoom_start=6)
    
    folium.Choropleth(
        geo_data=map_final,
        name="Choropleth",
        data=map_final,
        columns=["Aire_Sante", "Cas_Prevus"],
        key_on="feature.properties.Aire_Sante",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Cas de Rougeole Prévus"
    ).add_to(m)
    
    st_folium(m, width=700, height=500)

with col2:
    st.write("### Aires en Alerte")
    top_risk = risk_map_data.sort_values("Cas_Prevus", ascending=False).head(10)
    st.dataframe(top_risk)

    st.write("### Tendances")
    fig = px.line(future_df, x="Semaine", y="Cas_Prevus", color="Aire_Sante", title="Projection épidémique")
    st.plotly_chart(fig, use_container_width=True)
