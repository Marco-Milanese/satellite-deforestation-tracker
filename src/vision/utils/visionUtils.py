import torch
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dask.diagnostics import ProgressBar
from torchvision.transforms import GaussianBlur

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
    if hasattr(dataCube.data, 'dask'):
        print(f"Downloading and stitching tiles for bbox: {bbox}...")
        with ProgressBar():
            # .compute() triggers the actual download & stitch
            computed_data = dataCube.compute() 
    else:
        # Already computed (e.g. if you passed compute=True earlier)
        computed_data = dataCube

    date = pd.to_datetime(dataCube.time.values)
    year = date.year
    doy = date.dayofyear
    temporal_coords = torch.Tensor([[[year[-1], doy[-1] - 1]]])  # [1, 1, 2]
    print(F"DOY: {doy}, year: {year}")
    if year < 2022 or (year == 2022 and doy < 25):
        tensor = torch.from_numpy(dataCube.values).to(torch.float32) * 0.0001  # Scale 0-10000 to 0-1
    else:
        tensor = (torch.from_numpy(dataCube.values).to(torch.float32) * 0.0001) - 0.1   # Scale 0-10000 to 0-1
        
    tensor = tensor.unsqueeze(0)

    # Permute to Prithvi Order: (Batch, Channel, Time, Height, Width)
    # Current indices: 0=Batch, 1=Time, 2=Channel, 3=Height, 4=Width
    # Target indices:  0=Batch, 2=Channel, 1=Time, 3=Height, 4=Width
    tensor = tensor.permute(0, 2, 1, 3, 4)    
    
    
    
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    
    location_coords = torch.tensor([[center_lat, center_lon]], dtype=torch.float32)

    return {
        "pixel_values": tensor,           
        "temporal_coords": temporal_coords,
        "location_coords": location_coords
    }

def toGrid(embeddings, grid_size=14):
    """
    Converts embeddings to spatial grid. 
    Handles Batched Input: (B, 197, Embed_Dim) -> (B, Embed_Dim, grid_size, grid_size)
    """
    # Get last layer output: Shape (Batch, 197, Embedding_Dim)
    last_layer = embeddings[-1]
        
    B, N, C = last_layer.shape
    
    
    #Remove the CLS token (token that summarizes the whole image)
    CLStoken = last_layer[:, 0, :]
    no_cls = last_layer[:, 1:, :] #(B, 197, Embed_Dim) --> #(B, 197, Embed_Dim)
    
    #Trasform from list to grid
    grid_output = no_cls.view(B, grid_size, grid_size, C)
    
    #Reorganize dimentions
    grid_output = grid_output.permute(0, 3, 1, 2) # Shape: (B, C, 14, 14)

    return grid_output, CLStoken

def tensorToRGB(tensor):
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

def VisualizeComparison(beforeRGB, afterRGB, changeMap):
    # Handle tensor inputs for the map
    if isinstance(changeMap, torch.Tensor):
        heatmap_tensor = changeMap.detach().cpu()
        if heatmap_tensor.dim() == 4: # (B, C, H, W)
            heatmap_tensor = heatmap_tensor.squeeze(0) # (C, H, W)
    else:
        heatmap_tensor = torch.tensor(changeMap)
        
    # Ensure it's 4D for interpolation: (1, 1, H, W)
    if heatmap_tensor.dim() == 3:
        heatmap_tensor = heatmap_tensor.unsqueeze(0)
    elif heatmap_tensor.dim() == 2:
        heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)

    # Upscale Heatmap to match RGB image size
    heatmap_high_res = F.interpolate(
        heatmap_tensor, size=afterRGB.shape[:2], mode='bilinear', align_corners=False
    )
    #blurrer = GaussianBlur(kernel_size=21, sigma=5.0)
    #heatmap_high_res = blurrer(heatmap_high_res)
    heatmap_np = heatmap_high_res.squeeze().numpy()

    # --- AUTO-SCALE LOGIC ---
    # 1. Find the 98th percentile of the data (robust max)
    data_max = np.percentile(heatmap_np, 99)
    # 2. Set dynamic vmax:
    # Use data_max, but never go below 0.10 to prevent noise amplification
    dynamic_vmax = max(data_max, 0.10)
    dynamic_noise_threshold = dynamic_vmax * 0.2

    # --- VISUALIZATION SETUP ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    plt.subplots_adjust(bottom=0.25) 

    # Static Images
    axes[0].imshow(beforeRGB); axes[0].set_title("Before (T0)", fontsize=16); axes[0].axis('off')
    axes[1].imshow(afterRGB); axes[1].set_title("After (T1)", fontsize=16); axes[1].axis('off')
    
    # Dynamic Image (Overlay)
    axes[2].imshow(afterRGB)
    initial_mask = np.ma.masked_where(heatmap_np < dynamic_noise_threshold, heatmap_np)
    overlay = axes[2].imshow(
        initial_mask, cmap='RdYlGn_r', alpha=0.8, vmin=0, vmax=dynamic_vmax
    )
    axes[2].set_title("Deforestation Overlay", fontsize=16)
    axes[2].axis('off')

    # --- SLIDERS ---
    ax_sens = plt.axes([0.20, 0.1, 0.60, 0.03]) 
    slider_sens = Slider(ax_sens, 'Sensitivity (Max Color)', 0.05, 2.0, valinit=dynamic_vmax)

    ax_thres = plt.axes([0.20, 0.05, 0.60, 0.03]) 
    slider_thres = Slider(ax_thres, 'Noise Filter (Min Prob)', 0.0, 1.0, valinit=dynamic_noise_threshold)

    def update(val):
        overlay.set_clim(vmax=slider_sens.val)
        new_mask = np.ma.masked_where(heatmap_np < slider_thres.val, heatmap_np)
        overlay.set_data(new_mask)
        fig.canvas.draw_idle()

    slider_sens.on_changed(update)
    slider_thres.on_changed(update)

    plt.show()