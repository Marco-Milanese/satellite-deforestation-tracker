from src.vision.prithviInference import fullInference, tensorToRGB
from src.dataLoaders.sentinelDataFetcher import createDataCube, searchSTAC
from src.vision.sentinelDataTransform import cubeToPrithviFormat
from torch.functional import F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from math import cos

def cropCenter(image, target_size=224):
    """
    Takes an image of shape (H, W, C) and returns a center-cropped version 
    of shape (target_size, target_size, C).
    """
    h, w, c = image.shape
    
    # Only crop if the image is larger than the target
    if h > target_size or w > target_size:
        start_y = int(h / 2)
        start_x = int(w / 2)

        # Perform the slice
        cropped_image = image[start_y - int(target_size // 2) : start_y + int(target_size // 2), start_x - int(target_size // 2) : start_x + int(target_size // 2), :]
        return cropped_image
    
    # If image is smaller or equal, return original
    return image

def CosineSimilarityMap(before, after):
    
    similarityMap = F.cosine_similarity(before, after, dim=2)
    # 2. Convert to "Change Probability" (0 = No Change, 2 = Max Change)
    changeMap = 1 - similarityMap
    
    return changeMap

def cropPipeline(bbox, beforeDates, afterDates, model="prithvi_eo_v2_tiny_tl", visualize=False):
    # 1. Fetch DataCubes
    beforeItem = createDataCube(
        searchSTAC(beforeDates, bbox),
        bbox
    )
    afterItem = createDataCube(
        searchSTAC(afterDates, bbox),
        bbox
    )
    
    # 2. Convert to Prithvi Format Batches
    beforeBatch = cubeToPrithviFormat(beforeItem, bbox)
    afterBatch = cubeToPrithviFormat(afterItem, bbox)

    # 3. Run Inference
    beforeGrid = fullInference(beforeBatch, model=model)
    afterGrid = fullInference(afterBatch, model=model)

    beforeRGB = cropCenter(tensorToRGB(beforeBatch['pixel_values']))
    afterRGB = cropCenter(tensorToRGB(afterBatch['pixel_values']))
    print(f"Before RGB shape: {beforeRGB.shape}, After RGB shape: {afterRGB.shape}")

    # 4. Compute Change Map
    changeMap = CosineSimilarityMap(beforeGrid, afterGrid)
    
    if visualize:
            # Pre-calculate high-res heatmap ONCE
            heatmap_tensor = changeMap.unsqueeze(0).unsqueeze(0)
            heatmap_high_res = F.interpolate(
                heatmap_tensor, size=afterRGB.shape[:2], mode='bilinear', align_corners=False
            )
            heatmap_np = heatmap_high_res.squeeze().detach().cpu().numpy()

            # --- AUTO-SCALE LOGIC (The New Part) ---
            # 1. Find the 98th percentile of the data (robust max)
            data_max = np.percentile(heatmap_np, 98)
            
            # 2. Set dynamic vmax:
            # Use data_max, but never go below 0.20 (to prevent noise from looking like fire)
            dynamic_vmax = max(data_max, 0.10)
            dynamic_noise_threshold = dynamic_vmax * .17


            # --- VISUALIZATION SETUP ---
            fig, axes = plt.subplots(1, 3, figsize=(24, 9)) # Taller figure to fit sliders
            plt.subplots_adjust(bottom=0.25) # Make room at the bottom

            # Static Images
            axes[0].imshow(beforeRGB); axes[0].set_title("Before (T0)", fontsize=16); axes[0].axis('off')
            axes[1].imshow(afterRGB); axes[1].set_title("After (T1)", fontsize=16); axes[1].axis('off')
            
            # Dynamic Image (Overlay)
            axes[2].imshow(afterRGB)
            # Initialize with default threshold (0.05) and vmax (0.25)
            initial_mask = np.ma.masked_where(heatmap_np < dynamic_noise_threshold, heatmap_np)
            overlay = axes[2].imshow(
                initial_mask, cmap='RdYlGn_r', alpha=0.8, vmin=0, vmax=dynamic_vmax
            )
            axes[2].set_title("Deforestation Overlay", fontsize=16)
            axes[2].axis('off')

            # --- SLIDERS ---
            # Slider 1: Sensitivity (vmax) - Controls color saturation
            ax_sens = plt.axes([0.20, 0.1, 0.60, 0.03]) # [left, bottom, width, height]
            slider_sens = Slider(ax_sens, 'Sensitivity (Max Color)', 0.05, 1.0, valinit=dynamic_vmax)

            # Slider 2: Noise Filter (Threshold) - Controls what gets hidden
            ax_thres = plt.axes([0.20, 0.05, 0.60, 0.03]) 
            slider_thres = Slider(ax_thres, 'Noise Filter (Min Prob)', 0.0, 0.5, valinit=dynamic_noise_threshold)

            # Update Function
            def update(val):
                # 1. Update Color Limit (Sensitivity)
                overlay.set_clim(vmax=slider_sens.val)
                
                # 2. Update Mask (Threshold)
                # We must re-calculate the masked array based on the new slider value
                new_mask = np.ma.masked_where(heatmap_np < slider_thres.val, heatmap_np)
                overlay.set_data(new_mask)
                
                fig.canvas.draw_idle()

            # Link sliders to function
            slider_sens.on_changed(update)
            slider_thres.on_changed(update)

            plt.show()

    return beforeRGB, afterRGB, changeMap


import torch

def sliceTensor(tensor, target_size=224, stride=112):
    """
    Slices a (B, C, T, H, W) tensor into a list of (B, C, T, 224, 224) tensors
    using a sliding window.
    """
    # 1. Unfold Height (Dim 3)
    # Output: (B, C, T, n_h, W, patch_h)
    tiles = tensor.unfold(3, target_size, stride)
    
    # 2. Unfold Width (Dim 4 is now Dim 4 in original terms, but 4th index here)
    # Output: (B, C, T, n_h, n_w, patch_h, patch_w)
    tiles = tiles.unfold(4, target_size, stride)
    
    # 3. Permute to put spatial patches in a flat list order
    # Current: (B, C, T, n_h, n_w, patch_h, patch_w)
    # Target:  (n_h, n_w, B, C, T, patch_h, patch_w)
    tiles = tiles.permute(3, 4, 0, 1, 2, 5, 6)
    
    # 4. Flatten the patch grid (n_h * n_w) and convert to list
    # Shape: (Num_Patches, B, C, T, patch_h, patch_w)
    flat_tiles = tiles.reshape(-1, *tiles.shape[2:])
    
    # Convert to standard Python list
    return [t for t in flat_tiles]

# Usage
# large_tensor = torch.randn(1, 6, 8, 1000, 1000)
# tiles_list = slice_5d_tensor_overlapping(large_tensor, stride=112)

def stitchPipeline(bbox, beforeDates, afterDates, model="prithvi_eo_v2_tiny_tl", visualize=False, targetSize=224, overlapRatio=0.5):
    """
     Handles images bigger than 224*224 by cutting them up in suitable chunks that will be processed separately and then stitched back togheter
     
     bbox format: [minLon, minLat, maxLon, maxLat]

     1° lat --> 111 km
     1° lon --> 111*cos(lon) km
    
    """
    latDeltaKm = abs((bbox[3] - bbox[1]) * 111)
    lonDeltaKm = abs((bbox[0] - bbox[2]) * (111 * cos(bbox[0])))
    targetSizeKm = targetSize/100
    print(f"latDelta: {latDeltaKm}, lonDelta:{lonDeltaKm}")
    
    if(max(latDeltaKm, lonDeltaKm) < 1.5 * targetSizeKm):
       return cropPipeline(bbox, beforeDates, afterDates, model, visualize)
     
    """
     Cut the image in (h/targetSize)*(w/targetSize) tiles -->
     --> Pad tiles with h | w < targetSize with black border -->
     --> run inference and calculate heatmap --> stitch heatmap -->
     --> crop border

     tiles will be generated with an overlapRatio and then the values will be averaged when stitching.

     before tile (14, 14, 196) \
                                > similarity heatMap tile (14, 14) --> append to list of tiles -->
     after tile (14, 14, 196)  /
                                  --> merge one row, averaging the vertical overlap -->
                                  --> merge the rows averaging the horizontal overlap
    """
    beforeItem = createDataCube(
       searchSTAC(beforeDates, bbox),
       bbox)
     
    afterItem = createDataCube(
    searchSTAC(afterDates, bbox),
    bbox)
    
    #batch --> dict {
        #"pixel_values": tensor,   --> (Batch, Channel, Time, Height, Width)
        #"temporal_coords": temporal_coords,
        #"location_coords": location_coords
    #}
    beforeBatch = cubeToPrithviFormat(beforeItem, bbox)
    afterBatch = cubeToPrithviFormat(afterItem, bbox)

    beforeTiles = sliceTensor(beforeBatch["pixel_value"], targetSize, targetSize * overlapRatio)
    afterTiles = sliceTensor(afterBatch["pixel_value"], targetSize, targetSize * overlapRatio)



    """
    # 3. Run Inference
    beforeGrid = fullInference(beforeBatch, model=model)
    afterGrid = fullInference(afterBatch, model=model)

    beforeRGB = cropCenter(tensorToRGB(beforeBatch['pixel_values']))
    afterRGB = cropCenter(tensorToRGB(afterBatch['pixel_values']))
    print(f"Before RGB shape: {beforeRGB.shape}, After RGB shape: {afterRGB.shape}")
    """


# Tesla Giga Texas (Austin)
bbox_giga_tx = [-97.635, 30.210, -97.600, 30.240]

# Dates
date_before = "2020-05-01/2020-06-01" # Site Clearing Starting
date_after  = "2022-05-01/2022-06-01" # Fully Built
stitchPipeline(bbox_giga_tx, date_before, date_after, visualize=True)    



     