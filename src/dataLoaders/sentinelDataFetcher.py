import pystac_client
import planetary_computer
import stackstac
from itertools import groupby
import pandas as pd

URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_2A = "sentinel-2-l2a"
LANDSAT = "landsat-c2-l2"
PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]

def pcAuthenticator(url=URL):
    """Used to authenticate and access the planetary computer STAC API."""
    catalog = pystac_client.Client.open(
        url,
        modifier=planetary_computer.sign_inplace
    )
    return catalog

def searchSTAC(timeRange, bbox, catalog = pcAuthenticator(), collection = LANDSAT, cloudCoverThreshold=30, minCoverage=5, nImages = 5):
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
    print(f"{len(items)} tiles fetched")
    sorted_items = sorted(items, key=lambda x: x.properties.get("s2:mgrs_tile", "unknown"))

    best_items = []

    # 2. Group by Spatial Location
    for mgrs_id, group in groupby(sorted_items, key=lambda x: x.properties.get("s2:mgrs_tile", "unknown")):
        # 'group' is an iterator of all items in this specific grid cell (across different dates)
        tile_versions = list(group)

        valid_tiles = []
        for t in tile_versions:
            nodata = t.properties.get("s2:nodata_pixel_percentage", 0)
            valid_pct = 100 - nodata
            
            if valid_pct >= minCoverage:
                valid_tiles.append(t)
        # 3. Find the best one in this group
        # We find the item with the minimum 'eo:cloud_cover'.
        # We default to 100 if the property is missing to avoid crashes.
        sorted_tiles = sorted(valid_tiles, key=lambda x: x.properties.get("eo:cloud_cover", 100))
        
        # Take the top 2 (or fewer if less than 2 exist)
        top_tiles = sorted_tiles[:nImages]
        best_items.extend(top_tiles)
        
        # Optional: Print what we picked
        for t in top_tiles:
            print(f"Tile {mgrs_id}: Picked date {t.datetime.date()} with {t.properties['eo:cloud_cover']}% cloud coverage")
    #bestItem = min(items, key=lambda item: item.properties["eo:cloud_cover"])
    #dict_keys(['type', 'stac_version', 'stac_extensions', 'id', 'geometry', 'bbox', 'properties', 'links', 'assets', 'collection'])

    return best_items

def createLandsatDataCube(items, bbox, resolution=30):
    """
    Downloads and stitches Landsat 8/9 data.
    - Native Resolution: 30m
    - Scaling Formula: (DN * 2.75e-5) - 0.2
    """
    proj = items[0].properties["proj:code"]
    espgCode = int(proj.split(":")[-1])
    print(f"Using projection EPSG:{espgCode}")
    
    # 1. Define Landsat-Specific Assets (The 6 common bands)
    landsat_bands = ["blue", "green", "red", "nir08", "swir16", "swir22"]

    # 2. Calculate Representative Date (to fix the 1970 bug)
    dates = sorted([i.datetime for i in items])
    mid_date = dates[len(dates) // 2]
    representative_timestamp = pd.Timestamp(mid_date)
    print(f"Landsat Representative Date: {representative_timestamp.date()}")

    # 3. Create Stack
    # Note: We use epsg=4326 to match the Sentinel logic
    stack = stackstac.stack(
        items,
        assets=landsat_bands,
        bounds_latlon=bbox,
        resolution=resolution,
        epsg=espgCode, 
        fill_value=0
    )
    
    # 4. Convert to Float32
    stack = stack.astype("float32")

    # 5. Apply Landsat Collection 2 Scaling
    # Formula: reflectance = (DN * 0.0000275) - 0.2
    # We apply this to valid data only (not NaNs)
    #stack = (stack * 0.0000275) - 0.2

    # 6. Clip and Clean
    # Landsat valid range is 0.0 to 1.0, but atmospheric correction 
    # can sometimes result in slightly negative values or >1.0.
    stack = stack.clip(0.0, 1.0)
    
    # 7. Mosaic (Median to remove clouds)
    mosaic = stack.median(dim="time", keep_attrs=True)
    mosaic = mosaic.fillna(0.0)

    # 8. Re-attach Time
    mosaic = mosaic.expand_dims(time=[representative_timestamp])

    # 9. Compute
    # (Optional) with ProgressBar():
    mosaic = mosaic.compute()
    
    return mosaic

def createDataCube(item, bbox, bands = PRITHVI_BANDS, resolution=10, compute = False):

    proj = item[0].properties["proj:code"]
    espgCode = int(proj.split(":")[-1])
    print(f"Using projection EPSG:{espgCode}")

    dates = sorted([i.datetime for i in item])
    representative_date = dates[len(dates) // 2]
    print(f"Mosaic Representative Date: {representative_date.date()}")

    dataCube = stackstac.stack(
        item,
        assets = bands,
        bounds_latlon = bbox,
        resolution = resolution,
        epsg = espgCode,
        fill_value=0)
    

    mosaic = dataCube.median(dim="time", keep_attrs=True)

    # 3. Handle Missing Data
    # Fill any remaining gaps (e.g., if the tiles didn't fully cover the ROI)
    mosaic = mosaic.fillna(0)

    # 4. Add 'Time' dimension back 
    # (Models usually expect a Batch/Time dimension, e.g., (1, C, H, W))
    dataCube = mosaic.expand_dims(time=[pd.Timestamp(representative_date)])
    if compute:
        dataCube = dataCube.compute()
    
    return dataCube

"""
item = searchSTAC(("2023-01-01/2023-12-31"), [-122.5, 37.7, -122.3, 37.8])
dataCube = createDataCube(item, [-122.5, 37.7, -122.3, 37.8])
print(dataCube.sel(band="B04").isel(time=0))
"""
