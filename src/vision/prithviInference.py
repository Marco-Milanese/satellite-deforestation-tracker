"""Handles all the data preprocessing specific to Sentinel-2 imagery, for use with ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL model.

model input: 

forward(pixel_values (B, C, T, 224, 224), temporal_coord (B, T), location_coord (B, 2))

T --> number of temporal snapshots (time steps)
C --> number of spectral bands (channels)
Temporal coord: number between 1-366 representing day of year.
Location coord: latitude and longitude

Things to address:

Tiling strategy for large areas (beyond 224x224)
Normalizing pixel values to [0, 1]

"""

import torch
from terratorch.registry import BACKBONE_REGISTRY
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


import numpy as np
import torch

def tensorToRGB(tensor, target_size=224):
    """
    Converts Prithvi tensor to RGB with AUTOMATIC brightness normalization.
    - Handles [B, C, T, H, W], [C, H, W], etc.
    - Centers crop to target_size.
    - Automatically scales brightness using the 98th percentile.
    """
    # 1. Safety & Conversion
    if hasattr(tensor, "cpu"):
        image_data = tensor.detach().cpu().numpy()
    else:
        image_data = tensor.copy()

    # 2. Collapse Time & Batch Dimensions
    # Handle (1, 6, 8, 224, 224) -> Median over Time -> (1, 6, 224, 224)
    if image_data.ndim == 5:
        image_data = np.median(image_data, axis=2)
    
    # Handle (1, 6, 224, 224) -> (6, 224, 224)
    if image_data.ndim == 4:
        image_data = image_data.squeeze(0)

    # 3. Transpose to (H, W, Channels)
    if image_data.shape[0] == 6:
        image_data = np.transpose(image_data, (1, 2, 0))
    elif image_data.shape[-1] != 6:
        raise ValueError(f"Shape Error: Expected 6 channels. Got {image_data.shape}")

    # 4. Center Crop
    h, w, c = image_data.shape
    if target_size is not None and (h > target_size or w > target_size):
        start_y = (h - target_size) // 2
        start_x = (w - target_size) // 2
        image_data = image_data[start_y : start_y + target_size, start_x : start_x + target_size, :]

    # 5. Extract RGB (Blue=0, Green=1, Red=2)
    rgb = image_data[:, :, [2, 1, 0]] # Fancy indexing to reorder to R, G, B

    # --- AUTO-BRIGHTNESS LOGIC ---
    # We want the 98th percentile of the data to become "1.0" (White).
    # This ignores the top 2% of pixels (glare/clouds) to prevent them from ruining the contrast.
    
    p98 = np.percentile(rgb, 98)
    
    # Avoid division by zero if the image is empty
    if p98 > 0:
        scale_factor = 1.0 / p98
    else:
        scale_factor = 1.0 # Fallback
        
    rgb = rgb * scale_factor
    
    # Optional: Clip strictly to [0, 1]
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

def cropToTargetSize(batch, target_size=224):

    if batch['pixel_values'].shape[-1] > target_size:
        print(f"Cropping tensor from {batch['pixel_values'].shape} to {target_size}x{target_size}...")
        croppedBatch = batch.copy()
        croppedBatch["pixel_values"] = F.center_crop(batch['pixel_values'], output_size=[target_size, target_size])
        return croppedBatch

    
    print(f"No cropping needed, tensor shape: {batch['pixel_values'].shape}")

    return batch


def prithviInference(batch, model = "prithvi_eo_v2_tiny_tl"):
    """
    Given a Prithvi model and a data cube, performs inference to get embeddings.
    
    Args:
        model (torch.nn.Module): The Prithvi model loaded from BACKBONE_REGISTRY.
        dataCube (xr.DataArray): The data cube in xarray format.

    Returns:
        Model outputs a list of length: 12, each element is a tensor of shape torch.Size([1, 197, 192])
        Each element corresponds to features from a different layer in the backbone.
        192 is the embedding dimension, for each of the 197 patches (196 patches + 1 CLS token).
        CLS (index 0) represents information aggregated from the entire input image.
        All the other patches correspond to different spatial regions of the input image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_tiny_tl", pretrained=True
    )

    model.to(device)
    model.eval()

    with torch.no_grad():
        embeddings = model(    
        batch['pixel_values'],
        batch['temporal_coords'],
        batch['location_coords']
    )

    return embeddings

def toGrid(embeddings, grid_size=14):
    """
    Converts the last layer output embeddings to a spatial grid format.
    
    Args:
        embeddings (list): List of model outputs from prithviInference.
        grid_size (int): The height and width of the output grid (e.g., 14 for 14x14).
        
    Returns:
        torch.Tensor: A tensor of shape (grid_size, grid_size, Embedding_Dim).
    """
    lastLayerOutput = embeddings[-1].squeeze(0)  # Shape: (197, Embedding_Dim)
    lloNoCLS = lastLayerOutput[1:, :]  # Remove CLS token, Shape: (196, Embedding_Dim)
    gridOutput = lloNoCLS.view(grid_size, grid_size, -1)  # Shape: (grid_size, grid_size, Embedding_Dim)
    return gridOutput

def fullInference(batch, model = "prithvi_eo_v2_tiny_tl", grid_size=14):
    """
    Combines prithviInference and toGrid for end-to-end processing.
    
    Args:
        batch (dict): The input batch dictionary for the model.
        model (torch.nn.Module): The Prithvi model.
        grid_size (int): The height and width of the output grid.
        
    Returns:
        torch.Tensor: A tensor of shape (grid_size, grid_size, Embedding_Dim).
    """
    cropped = cropToTargetSize(batch)
    embeddings = prithviInference(cropped, model)
    gridOutput = toGrid(embeddings, grid_size)
    
    return gridOutput