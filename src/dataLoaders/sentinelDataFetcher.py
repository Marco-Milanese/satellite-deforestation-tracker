"""
Defines functions to fetch and preprocess Sentinel satellite data for analysis.

sentinelDataFetcher(startDate, endDate, region) --> Fetches sentinel data of the specified region around the before and after dates.
                                                    Uses microsoft STAC API to fetch the data.
                                                    Filters images based on cloud coverage (least cloudy images are selected).
                                                    Returns a list of image URLs for further processing.

"""