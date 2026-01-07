from src.vision.prithviInference import fullInference
from src.vision.utils.visionUtils import tensorToRGB, VisualizeComparison, fetchAndConvert
import torch
import torch.nn.functional as F
from math import floor
from terratorch.registry import BACKBONE_REGISTRY
from tqdm import tqdm

def CosineSimilarityMap(before, after):
    """
    Calculates Cosine changeMapChunkilarity along the channel dimension (dim=1).
    
    Args:
        before: Tensor of shape (Batch, 192, 14, 14)
        after:  Tensor of shape (Batch, 192, 14, 14)
        
    Returns:
        Tensor of shape (Batch, 1, 14, 14) representing change probability.
    """
    changeMap = 1 - F.cosine_similarity(before, after, dim=1)

    return changeMap.unsqueeze(1)


def stitchPipeline(bbox, batches, model="prithvi_eo_v2_tiny_tl", gridSize=14, targetSize=224, overlapRatio=0.5, chunkSize=16):
    """
    Optimized pipeline with SHAPE SAFEGUARDS for multi-sensor/multi-year data.
    """
    print(f"Analysing area: [{bbox}]")
    
    beforeBatch = batches[0]
    afterBatch = batches[1]
    
    # Extracting pixel values
    beforePixelValues = beforeBatch["pixel_values"] 
    afterPixelValues = afterBatch["pixel_values"]
    
    # 1. Capture Reference Dimensions from "Before" Image
    B, C, T, H, W = beforePixelValues.shape 
    device = beforePixelValues.device
    print(f"Reference Image Shape: {H}x{W}")

    # --- FIX: SHAPE HARMONIZATION ---
    # Check if "After" image matches "Before" image dimensions exactly
    if afterPixelValues.shape[-2:] != (H, W):
        print(f"Shape Mismatch Detected! Resizing 'After' image from {afterPixelValues.shape[-2:]} to {(H, W)}")
        
        # Collapse dimensions for interpolation: (B, C*T, H_old, W_old)
        Ba, Ca, Ta, Ha, Wa = afterPixelValues.shape
        afterFlat = afterPixelValues.view(Ba, Ca * Ta, Ha, Wa)
        
        # Interpolate to match Reference (H, W)
        afterResized = F.interpolate(
            afterFlat, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Restore dimensions: (B, C, T, H, W)
        afterPixelValues = afterResized.view(B, C, T, H, W)
    # --------------------------------

    # 2. Calculate Stride and Padding
    inverseOverlapRatio = 1 - overlapRatio
    inputStride = int(floor(targetSize * inverseOverlapRatio))
    scaleFactor = targetSize // gridSize
    outputStride = inputStride // scaleFactor
    
    padH = (inputStride - (H - targetSize) % inputStride) % inputStride
    padW = (inputStride - (W - targetSize) % inputStride) % inputStride
    
    beforePixelValues = F.pad(beforePixelValues, (0, padW, 0, padH))
    afterPixelValues = F.pad(afterPixelValues, (0, padW, 0, padH))
    
    hPadded, wPadded = beforePixelValues.shape[-2:]
    print(f"Padded Image Shape: {hPadded}x{wPadded} (Padding: H={padH}, W={padW})")

    # 3. Unfold Pixels (Create Tiles)
    # Now this is safe because both tensors are guaranteed to be hPadded x wPadded
    beforeReshaped = beforePixelValues.view(B, C * T, hPadded, wPadded)
    afterReshaped = afterPixelValues.view(B, C * T, hPadded, wPadded)

    beforeUnfolded = F.unfold(beforeReshaped, kernel_size=targetSize, stride=inputStride)
    afterUnfolded = F.unfold(afterReshaped, kernel_size=targetSize, stride=inputStride)

    patchesNumber = beforeUnfolded.shape[-1]
    
    # ... (Rest of your function remains exactly the same) ...
    
    # Reshape: (Total_Tiles, C, T, H_tile, W_tile)
    beforeTiles = beforeUnfolded.transpose(1, 2).reshape(-1, C, T, targetSize, targetSize)
    afterTiles = afterUnfolded.transpose(1, 2).reshape(-1, C, T, targetSize, targetSize)

    # Generate Correct Metadata (Lat/Lon) per Tile
    nRows = (hPadded - targetSize) // inputStride + 1
    nCols = (wPadded - targetSize) // inputStride + 1
    
    latCenters = torch.linspace(bbox[3], bbox[1], nRows, device=device)
    lonCenters = torch.linspace(bbox[0], bbox[2], nCols, device=device)
    
    gridLat, gridLon = torch.meshgrid(latCenters, lonCenters, indexing='ij')
    
    flatLats = gridLat.flatten()
    flatLons = gridLon.flatten()
    locTiles = torch.stack((flatLats, flatLons), dim=1)
    
    if B > 1: locTiles = locTiles.repeat(B, 1)

    # Extract temporal coords
    beforeTempCoords = beforeBatch["temporal_coords"]
    afterTempCoords = afterBatch["temporal_coords"]
    
    # Repeat for every patch
    beforeTempTiles = beforeTempCoords.repeat_interleave(patchesNumber, dim=0)
    afterTempTiles = afterTempCoords.repeat_interleave(patchesNumber, dim=0)

    # Batched Inference Loop
    heatmapPatches = []
    totalTiles = beforeTiles.shape[0]

    print(f"Running inference on {totalTiles} tiles. Batch size: {chunkSize}")

    for i in tqdm(range(0, totalTiles, chunkSize)):
        # Slice chunks
        bChunk = beforeTiles[i:i+chunkSize]
        aChunk = afterTiles[i:i+chunkSize]
        locChunk = locTiles[i:i+chunkSize]
        
        # Get specific temporal coords for this chunk
        beforeTempChunk = beforeTempTiles[i:i+chunkSize]
        afterTempChunk = afterTempTiles[i:i+chunkSize]
        
        # Construct batches with CORRECT distinct dates
        beforeTileBatch = {
            "pixel_values": bChunk, 
            "temporal_coords": beforeTempChunk, 
            "location_coords": locChunk
        }
        afterTileBatch = {
            "pixel_values": aChunk, 
            "temporal_coords": afterTempChunk, 
            "location_coords": locChunk
        }
        with torch.no_grad():
            beforeGrid = fullInference(beforeTileBatch, model=model)
            afterGrid = fullInference(afterTileBatch, model=model)
            
            # Calculate changeMapChunkilarity
            changeMapChunk = CosineSimilarityMap(beforeGrid, afterGrid)
        
        heatmapPatches.append(changeMapChunk)

    # Concatenate: (Total_Tiles, 1, 14, 14)
    allHeatmaps = torch.cat(heatmapPatches, dim=0)

    # Stitching (Fold)
    patchesToFold = allHeatmaps.view(B, patchesNumber, -1).transpose(1, 2)

    outH = hPadded // scaleFactor
    outW = wPadded // scaleFactor
    
    # Fold Sum
    stitchedSum = F.fold(patchesToFold, output_size=(outH, outW), kernel_size=gridSize, stride=outputStride)

    # Fold Count
    onesPatches = torch.ones_like(patchesToFold)
    overlapCounter = F.fold(onesPatches, output_size=(outH, outW), kernel_size=gridSize, stride=outputStride)
    
    finalHeatmap = stitchedSum / (overlapCounter + 1e-8)

    # Crop & Return
    validH = H // scaleFactor
    validW = W // scaleFactor
    finalHeatmap = finalHeatmap[:, :, :validH, :validW]

    beforeRGB = tensorToRGB(beforePixelValues[:, :, :, :H, :W])
    afterRGB = tensorToRGB(afterPixelValues[:, :, :, :H, :W])
    
    print("Stitching complete.")
    return beforeRGB, afterRGB, finalHeatmap

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    """
    #bbox_giga_tx = [-97.670, 30.180, -97.565, 30.270]
    #date_before = "2020-05-01/2020-06-01"
    #date_after  = "2022-05-01/2022-06-01"
    
    bbox_istanbul_airport = [28.600, 41.200, 28.900, 41.350]

    # Dates
    date_before = "2015-06-01/2015-08-01"  # Early Construction (Mostly Forest/Soil)
    date_after  = "2022-06-01/2022-08-01"  # Fully Operational (Concrete/Terminals)
    """
    
   
    # Lake Aculeo (Santiago, Chile)
    # A tourist lake that turned into a dusty plain.
    bbox_lake_aculeo = [-70.920, -33.850, -70.850, -33.800]

    # Before: Water still present (Sentinel-2A Launch era)
    date_before = "2016-01-01/2016-04-01"

    # After: Completely Dry
    date_after  = "2019-01-01/2019-04-01"

    #bbox = [31.50, 23.00, 31.90, 23.50]
    #date_before = "1998-01-01/1998-04-01"
    #date_after = "2002-01-01/2002-04-01"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl", pretrained=True)
    model.to(device)

    batches = fetchAndConvert(bbox_lake_aculeo, date_before, date_after, "LANDSAT")

    beforeRGB, afterRGB, changeMap = stitchPipeline(bbox_lake_aculeo, batches, model=model, overlapRatio=0.5)
    VisualizeComparison(beforeRGB, afterRGB, changeMap)