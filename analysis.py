from src.vision.prithviInference import fullInference, tensorToRGB
from src.dataLoaders.sentinelDataFetcher import createDataCube, searchSTAC
from src.vision.sentinelDataTransform import cubeToPrithviFormat
from torch.functional import F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from math import cos, floor
import torch

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
    changeMapShape = changeMap.shape
    
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

def stitchPipeline(bbox, beforeDates, afterDates, model="prithvi_eo_v2_tiny_tl", visualize=False, targetSize=224, overlapRatio=0.5):
    """
     Handles images bigger than 224*224 by cutting them up in suitable chunks that will be processed separately and then stitched back togheter
     
     bbox format: [minLon, minLat, maxLon, maxLat]

     1° lat --> 111 km
     1° lon --> 111*cos(lon) km
    
    """
    """
     Cut the image in (h/targetSize)*(w/targetSize) tiles with an overlap of overlapRatio % -->
     --> run inference and calculate heatmap --> stitch heatmap
     
     If the image size is notperfectly divisible by the target size partial tiles will be ignored

     tiles will be generated with an overlapRatio and then the values will be averaged when stitching.

     before tile (14, 14, 196) 
                                > similarity heatMap tile (14, 14) --> append to list of tiles -->
     after tile (14, 14, 196)  /
                                  --> merge one row, averaging the vertical overlap -->
                                  --> merge the rows averaging the horizontal overlap
    """
    latDeltaKm = abs((bbox[3] - bbox[1]) * 111)
    lonDeltaKm = abs((bbox[0] - bbox[2]) * (111 * cos(bbox[0])))
    targetSizeKm = targetSize/100
    print(f"latDelta: {latDeltaKm}, lonDelta:{lonDeltaKm}")
    
    if(max(latDeltaKm, lonDeltaKm) < 2 * targetSizeKm):
       print("Cropping...")
       return cropPipeline(bbox, beforeDates, afterDates, model, visualize)
    
    
    print("Stitching...")
    beforeItem = createDataCube(
       searchSTAC(beforeDates, bbox),
       bbox)
     
    afterItem = createDataCube(
    searchSTAC(afterDates, bbox),
    bbox)
    
    beforeBatch = cubeToPrithviFormat(beforeItem, bbox)
    afterBatch = cubeToPrithviFormat(afterItem, bbox)
    #batch --> dict {
        #"pixel_values": tensor,   --> (Batch, Channel, Time, Height, Width)
        #"temporal_coords": temporal_coords,
        #"location_coords": location_coords
    #}
    
    stride = int(floor(targetSize * overlapRatio))
    h, w = beforeBatch["pixel_values"].shape[-2:]
    newH = targetSize * (floor(h/targetSize))
    newW = targetSize * (floor(w/targetSize))

    beforePixelValues = beforeBatch["pixel_values"][:, :, :, :newH, :newW]
    afterPixelValues = afterBatch["pixel_values"][:, :, :, :newH, :newW]

    beforePixelValuesShape = beforePixelValues.shape

    columns = floor(newW / stride) - 1
    rows = floor(newH / stride) - 1

    #beforeTiles = sliceTensor(beforeBatch["pixel_values"], targetSize, targetSize * overlapRatio)
    #afterTiles = sliceTensor(afterBatch["pixel_values"], targetSize, targetSize * overlapRatio)
    
    heatMapTiles = torch.zeros(rows, columns, 14, 14)
    #0=Batch, 2=Channel, 1=Time, 3=Height, 4=Width
    for i in range(rows):
        for j in range(columns):
            #(Batch, Channel, Time, Height, Width)
            #start : stop : step
            hStart = i * stride
            hStop = hStart + targetSize
            wStart = j * stride
            wStop = wStart + targetSize
            beforeTile = beforePixelValues[:, :, :, hStart:hStop, wStart:wStop]
            afterTile = afterPixelValues[:, :, :, hStart:hStop, wStart:wStop]
            beforeTileShape = beforeTile.shape
            afterTileShape = afterTile.shape

            beforeTileBatch = {
                "pixel_values": beforeTile,           
                "temporal_coords": beforeBatch["temporal_coords"],
                "location_coords": beforeBatch["location_coords"]
            }
            afterTileBatch = {
                "pixel_values": afterTile,           
                "temporal_coords": afterBatch["temporal_coords"],
                "location_coords": afterBatch["location_coords"]
            }

            beforeTileGrid = fullInference(beforeTileBatch, model=model)
            afterTileGrid = fullInference(afterTileBatch, model=model)

            heatMapTiles[i, j] = (CosineSimilarityMap(beforeTileGrid, afterTileGrid))

    heatMapTilesShape = heatMapTiles.shape
    mergedHeatMap = torch.empty(0, int(((columns + 1)/2) * 14))
    for i in range(rows):
        mergedHeatMapRow = torch.empty(14, 0)
        for j in range(columns):
            currentTile = heatMapTiles[i, j]
            if not (j == 0 or j == columns - 1):
                print("j != 0 & j != columns - 1")
                previousTile = heatMapTiles[i, j - 1]
                nextTile = heatMapTiles[i, j + 1]
                firstHalfMerged = (previousTile[..., 7:] + currentTile[..., :7]) / 2
            elif(j == 0):
                print("j == 0")
                nextTile = heatMapTiles[i, j + 1]
                firstHalfMerged = currentTile[..., :7]
                secondHalfMerged = (currentTile[..., :7] + nextTile[..., 7:]) / 2
            else:
                print("j == columns - 1")
                previousTile = heatMapTiles[i, j - 1]
                firstHalfMerged = (previousTile[..., 7:] + currentTile[..., :7]) / 2
                secondHalfMerged = currentTile[..., 7:]
                
            mergedTile = torch.cat((firstHalfMerged, secondHalfMerged), dim=1)

            mergedHeatMapRow = torch.cat((mergedHeatMapRow[..., :-7], mergedTile), dim=1)
                
        mergedHeatRowShape = mergedHeatMapRow.shape
        print("Merging Row...")
        mergedHeatMap = torch.cat((mergedHeatMap[:-7,...], mergedHeatMapRow), dim=0)
        mergedHeatMapShape = mergedHeatMap.shape
    

    beforeRGB = tensorToRGB(beforePixelValues)
    beforeRGBShape = beforeRGB.shape
    afterRGB = tensorToRGB(afterPixelValues)
    afterRGBShape = afterRGB.shape
    changeMap = mergedHeatMap
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

        plt.show(block=True)

    return beforeRGB, afterRGB, changeMap

# Tesla Giga Texas (Austin)
bbox_giga_tx = [-97.670, 30.180, -97.565, 30.270]

# Dates
date_before = "2020-05-01/2020-06-01" # Site Clearing Starting
date_after  = "2022-05-01/2022-06-01" # Fully Built
cropPipeline(bbox_giga_tx, date_before, date_after, visualize=True)    



     