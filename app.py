import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(page_title="Surveillance Rougeole Multi-pays", layout="wide", page_icon="🦠")
st.title("🦠 Dashboard de Surveillance Prédictive – Rougeole")

# =======================
# 1. Sidebar – Sélections
# =======================
pays_selectionne = st.sidebar.selectbox("Pays", ["Niger","Burkina Faso","Mali"])
option_aire = st.sidebar.radio("Source Aires de Santé", ["Upload Shapefile/GeoJSON"])
upload_file = st.sidebar.file_uploader("Shapefile/GeoJSON", type=["shp","geojson"])

linelist_file = st.sidebar.file_uploader("Linelists CSV", type=["csv"])
vacc_file = st.sidebar.file_uploader("Vaccination CSV (optionnel)", type=["csv"])
start_date = st.sidebar.date_input("Date début", datetime(2024,1,1))
end_date = st.sidebar.date_input("Date fin", datetime.today())

# =======================
# 2. Aires de santé
# =======================
if upload_file:
    sa_gdf = gpd.read_file(upload_file)
    sa_gdf["ADM3_NAME"] = sa_gdf["ADM3_NAME"].astype(str)
else:
    st.warning("Uploader un fichier pour continuer")
    st.stop()

aires_list = sa_gdf['ADM3_NAME'].tolist()

# =======================
# 3. Linelist
# =======================
@st.cache_data
def generate_dummy_linelists(n=400, aires=[]):
    np.random.seed(42)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.random.randint(0,180,n), unit="D")
    return pd.DataFrame({
        "ID_Cas": range(1,n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(np.random.randint(1,5,n), unit="D"),
        "Aire_Sante": np.random.choice(aires,n),
        "Age_Mois": np.random.randint(6,180,n),
        "Statut_Vaccinal": np.random.choice(["Oui","Non"], n, p=[0.6,0.4])
    })

if linelist_file:
    df = pd.read_csv(linelist_file, parse_dates=["Date_Debut_Eruption","Date_Notification"])
else:
    df = generate_dummy_linelists(aires=aires_list)

df = df[(df["Date_Debut_Eruption"]>=pd.to_datetime(start_date)) & (df["Date_Debut_Eruption"]<=pd.to_datetime(end_date))]
df["Semaine"] = df["Date_Debut_Eruption"].dt.to_period("W").astype(str)

# =======================
# 4. Préparation features et prédiction
# =======================
weekly_features = df.groupby(["Aire_Sante","Semaine"]).agg(
    Cas_Observes=("ID_Cas","count"),
    Non_Vaccines=("Statut_Vaccinal", lambda x: (x=="Non").mean()*100)
).reset_index()

weekly_features["Pop_0_4"] = np.random.randint(500,5000,len(weekly_features))
weekly_features["Urban_Encoded"] = np.random.randint(0,2,len(weekly_features))

feature_cols = ["Cas_Observes","Non_Vaccines","Pop_0_4","Urban_Encoded"]
X = weekly_features[feature_cols]
y = weekly_features["Cas_Observes"]

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X,y)

# Prévision 12 semaines
future_weeks = []
n_weeks = 12
latest_week_idx = len(weekly_features["Semaine"].unique())
for aire in weekly_features["Aire_Sante"].unique():
    aire_row = weekly_features[weekly_features["Aire_Sante"]==aire].iloc[-1]
    for i in range(1,n_weeks+1):
        future_weeks.append({
            "Aire_Sante":aire,
            "Semaine":f"Week_{latest_week_idx+i}",
            "Cas_Observes":aire_row["Cas_Observes"],
            "Non_Vaccines":aire_row["Non_Vaccines"],
            "Pop_0_4":aire_row["Pop_0_4"],
            "Urban_Encoded":aire_row["Urban_Encoded"]
        })

future_df = pd.DataFrame(future_weeks)
future_df["Predicted_Cases"] = model.predict(future_df[feature_cols])

risk_df = future_df.groupby("Aire_Sante").agg(
    Max_Predicted_Cases=("Predicted_Cases","max"),
    Week_of_Peak=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(),"Semaine"])
).reset_index()

# =======================
# 5. Carte Folium enrichie
# =======================
st.subheader("🗺️ Carte – Risque rougeole")

m = folium.Map(location=[15,8], zoom_start=6)
sa_merge = sa_gdf.merge(risk_df,left_on="ADM3_NAME",right_on="Aire_Sante",how="left")

colormap = cm.linear.OrRd_09.scale(0, sa_merge["Max_Predicted_Cases"].max())
colormap.caption = "Cas rouges sur 12 semaines"
colormap.add_to(m)

for _, row in sa_merge.iterrows():
    popup_text = f"""
    <b>{row['ADM3_NAME']}</b><br>
    Cas Observés: {int(weekly_features[weekly_features['Aire_Sante']==row['ADM3_NAME']]['Cas_Observes'].sum())}<br>
    Cas Prévus: {int(row['Max_Predicted_Cases'])}<br>
    % Non vaccinés: {weekly_features[weekly_features['Aire_Sante']==row['ADM3_NAME']]['Non_Vaccines'].mean():.1f}%<br>
    Population 0-4 ans: {int(weekly_features[weekly_features['Aire_Sante']==row['ADM3_NAME']]['Pop_0_4'].mean())}<br>
    Semaine de pic: {row['Week_of_Peak']}
    """
    folium.GeoJson(
        row['geometry'],
        style_function=lambda f, color=colormap(row['Max_Predicted_Cases']): {'fillColor': color,'color':'black','weight':1,'fillOpacity':0.7},
        tooltip=popup_text
    ).add_to(m)

st_folium(m,width=900,height=650)

# =======================
# 6. Courbes Observé vs Prévu
# =======================
st.subheader("📈 Courbes Observé vs Prévu")
plot_df = pd.concat([
    weekly_features[["Semaine","Cas_Observes","Aire_Sante"]],
    future_df.rename(columns={"Predicted_Cases":"Cas_Prevus"})[["Semaine","Cas_Prevus","Aire_Sante"]]
], axis=0)
fig_obs = px.line(plot_df, x="Semaine", y="Cas_Observes", color="Aire_Sante", labels={"Cas_Observes":"Cas Observés"})
fig_pred = px.line(plot_df, x="Semaine", y="Cas_Prevus", color="Aire_Sante", labels={"Cas_Prevus":"Cas Prévus"})
st.plotly_chart(fig_obs,use_container_width=True)
st.plotly_chart(fig_pred,use_container_width=True)

# =======================
# 7. KPI – Statistiques
# =======================
st.subheader("📊 KPI – Indicateurs par Aire de Santé")
kpi_df = weekly_features.groupby("Aire_Sante").agg(
    Cas_Observes=("Cas_Observes","sum"),
    Cas_Prevus=("Aire_Sante", lambda x: int(risk_df[risk_df['Aire_Sante']==x.iloc[0]]['Max_Predicted_Cases'])),
    Non_Vaccines_Pct=("Non_Vaccines","mean"),
    Pop_0_4=("Pop_0_4","mean")
).reset_index()
kpi_df["Non_Vaccines_Pct"] = kpi_df["Non_Vaccines_Pct"].round(1)

st.dataframe(kpi_df.sort_values("Cas_Prevus",ascending=False))

# Alertes
alert_df = kpi_df[kpi_df["Cas_Observes"]>=kpi_df["Cas_Observes"].quantile(0.75)]
st.subheader("🚨 Aires de Santé en Alerte Rouge")
st.dataframe(alert_df)
