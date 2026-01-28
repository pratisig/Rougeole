# gee_utils.py
import ee
import geopandas as gpd

@st.cache_resource
def init_gee():
    key = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
    creds = ee.ServiceAccountCredentials(
        key["client_email"], key_data=json.dumps(key)
    )
    ee.Initialize(creds)
    return True

gee_ok = init_gee()

def gdf_to_ee(gdf):
    return ee.FeatureCollection(gdf.__geo_interface__)
# gee_utils.py
def worldpop_by_age(ee_fc):
    dataset = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")

    age_bands = {
        "under1": ["f_00", "m_00"],
        "1_4": ["f_01", "m_01"],
        "5_9": ["f_05", "m_05"]
    }

    results = {}

    for group, bands in age_bands.items():
        img = dataset.select(bands).sum()

        stats = img.reduceRegions(
            collection=ee_fc,
            reducer=ee.Reducer.mean(),
            scale=100
        )

        results[group] = stats

    return results
# gee_utils.py
def urban_ratio(ee_fc):
    ghsl = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0")

    urban = ghsl.eq(2).Or(ghsl.eq(3))

    stats = urban.reduceRegions(
        collection=ee_fc,
        reducer=ee.Reducer.mean(),
        scale=100
    )

    return stats
