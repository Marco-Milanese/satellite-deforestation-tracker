import pystac_client
import planetary_computer
import stackstac
import rasterio

URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_2A = "sentinel-2-l2a"
PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]

def pcAuthenticator(url=URL):
    """Used to authenticate and access the planetary computer STAC API."""
    catalog = pystac_client.Client.open(
        url,
        modifier=planetary_computer.sign_inplace
    )
    return catalog

def searchSTAC(timeRange, bbox, catalog = pcAuthenticator(), collection = SENTINEL_2A, cloudCoverThreshold=25):
    """Searches the STAC API for Sentinel-2 L2A images within the specified time range and bounding box.
    
    Args:
        catalog: The STAC catalog to search.
        timeRange: A tuple of start and end dates in 'YYYY-MM-DD' format.
        bbox: A list defining the bounding box [minLon, minLat, maxLon, maxLat].
        cloudCoverThreshold: Maximum acceptable cloud cover percentage.

        returns: The STAC item with the least cloud cover within the specified parameters.
    """
    search = catalog.search(collections = [collection], datetime = timeRange, bbox = bbox, query = {"eo:cloud_cover": {"lt": cloudCoverThreshold}})

    items = search.get_all_items()

    bestItem = min(items, key=lambda item: item.properties["eo:cloud_cover"])
    #dict_keys(['type', 'stac_version', 'stac_extensions', 'id', 'geometry', 'bbox', 'properties', 'links', 'assets', 'collection'])

    return bestItem

def createDataCube(item, bbox, bands = PRITHVI_BANDS, resolution=10, compute = False):

    proj = item.properties["proj:code"]
    espgCode = int(proj.split(":")[-1])
    print(f"Using projection EPSG:{espgCode}")

    dataCube = stackstac.stack(
        item,
        assets = bands,
        bounds_latlon = bbox,
        resolution = resolution,
        epsg = espgCode,
        fill_value=0)
    
    if compute:
        dataCube = dataCube.compute()
    
    return dataCube

"""
item = searchSTAC(("2023-01-01/2023-12-31"), [-122.5, 37.7, -122.3, 37.8])
dataCube = createDataCube(item, [-122.5, 37.7, -122.3, 37.8])
print(dataCube.sel(band="B04").isel(time=0))
"""
