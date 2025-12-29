import torch
from src.vision.utils.visionUtils import toGrid


def prithviInference(batch, model):
    """
    Runs model inference on a batch.
    Returns the list of embeddings. 
    """
    device = next(model.parameters()).device # Get model's device
    model.eval()

    # Move inputs to the correct device
    pixel_values = batch['pixel_values'].to(device)
    temporal_coords = batch['temporal_coords'].to(device)
    location_coords = batch['location_coords'].to(device)

    with torch.no_grad():
        embeddings = model(    
            pixel_values,
            temporal_coords,
            location_coords
        )

    return embeddings

def fullInference(batch, model, grid_size=14):
    """
    End-to-end inference pipeline.
    """
    embeddings = prithviInference(batch, model) #Runs inference on a batch of tiles, returns a list of tokens
    gridOutput, clsToken = toGrid(embeddings, grid_size) 
    
    return gridOutput