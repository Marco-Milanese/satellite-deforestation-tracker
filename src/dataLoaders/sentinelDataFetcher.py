import pystac_client
import planetary_computer
import stackstac
import rasterio

URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_2A = "sentinel-2-l2a"

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

def createDataCube(item, bbox, bands = ["B02", "B03", "B04", "B8A", "B11", "B12"], resolution=10, compute = True):

    dataCube = stackstac.stack(
        item,
        assets = bands,
        bounds_latlon = bbox,
        resolution = resolution,
        epsg = 4326)
    
    if compute:
        dataCube = dataCube.compute()
    
    return dataCube
