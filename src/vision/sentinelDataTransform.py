import torch
import numpy as np
import pandas as pd

def cubeToPrithviFormat(dataCube, bbox):
    """
    Converts the stackstac DataCube into the exact dictionary structure 
    required by Prithvi / TerraTorch.
    
    Args:
        data_cube (xr.DataArray): The output from createDataCube.
                                  Shape: (Time, Band, Y, X)
        bbox (list): [min_lon, min_lat, max_lon, max_lat] for location embedding.
        
    Returns:
        dict: A batch dictionary ready for 'model(batch)'.
    """
    
    # 1. PIXEL VALUES: Shape (Batch, Channel, Time, Height, Width)
    # ------------------------------------------------------------
    # Current xarray shape: (Time, Band, Y, X) -> e.g. (1, 6, 224, 224)
    
    # Convert to numpy and then torch
    # We use .values to strip xarray metadata
    tensor = torch.from_numpy(dataCube.values) * 0.0001  # Scale 0-10000 to 0-1
    # Add Batch Dimension -> (1, Time, Band, Y, X)
    tensor = tensor.unsqueeze(0)
    # Permute to Prithvi Order: (Batch, Channel, Time, Height, Width)
    # Current indices: 0=Batch, 1=Time, 2=Channel, 3=Height, 4=Width
    # Target indices:  0=Batch, 2=Channel, 1=Time, 3=Height, 4=Width
    tensor = tensor.permute(0, 2, 1, 3, 4)
    # Validation: Ensure it is Float32 and normalized 0-1
    # stackstac(rescale=True) gives us 0.0-1.0 range, but check dtype.
    tensor = tensor.to(torch.float32)

    # 2. TEMPORAL COORDS: Shape (Batch, Time)
    # ---------------------------------------
    # Extract dates from xarray coordinates
    # We need "Day of Year" (1-365)
    
    
    date = pd.to_datetime(dataCube.time.values)
    year = date.year
    doy = date.dayofyear
    temporal_coords = torch.Tensor([[[year[-1], doy[-1] - 1]]])  # [1, 1, 2]


    # 3. LOCATION COORDS: Shape (Batch, 2)
    # ------------------------------------
    # Center Lat/Lon of the bounding box
    
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    
    location_coords = torch.tensor([[center_lat, center_lon]], dtype=torch.float32)

    return {
        "pixel_values": tensor,           
        "temporal_coords": temporal_coords,
        "location_coords": location_coords
    }