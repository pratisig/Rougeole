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
