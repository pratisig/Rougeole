# viz_utils.py
import folium

def risk_map(gdf):
    m = folium.Map(location=[13.5, -1.5], zoom_start=6)

    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=["name", "risk"],
        key_on="feature.properties.name",
        fill_color="YlOrRd"
    ).add_to(m)

    return m
