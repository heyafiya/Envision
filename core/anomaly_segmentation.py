import os
import torch
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from core.models.autoencoder import EnhancementAutoencoder
from core.io_utils import get_middle_slice
import nibabel as nib

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_enhancement_model(region, weight_dir="core/outputs/weights"):
    device = get_device()
    model = EnhancementAutoencoder().to(device)
    weight_path = os.path.join(weight_dir, f"{region.lower()}_autoencoder.pt")
    
    if os.path.exists(weight_path):
        try:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except Exception as e:
            print(f"Warning: Failed to load {weight_path}. Using random initialization. {e}")
    else:
        print(f"Warning: {weight_path} not found. Using untrained model for demo purposes.")
        
    model.eval()
    return model

def compute_anomaly_map(model, input_2d):
    """
    input_2d: numpy array of shape (224, 224) normalized to [0, 1]
    returns: (reconstruction_2d, error_heatmap)
    """
    device = get_device()
    input_tensor = torch.from_numpy(input_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        recon_tensor = model(input_tensor)
        
    recon_2d = recon_tensor.squeeze().cpu().numpy()
    
    # Reconstruction error (Absolute Difference)
    error_map = np.abs(input_2d - recon_2d)
    
    # Smooth the error map to remove high-frequency noise/edge artifacts
    error_map = cv2.GaussianBlur(error_map, (5, 5), 0)
    
    # Normalize error map to [0, 1] for heatmap visualization
    m, M = error_map.min(), error_map.max()
    if M > m:
        error_heatmap = (error_map - m) / (M - m)
    else:
        error_heatmap = error_map
        
    return recon_2d, error_heatmap

def apply_threshold(error_heatmap, threshold_override=None, input_2d=None, postprocess=True):
    """
    Applies Otsu or manual threshold to get binary ROI mask.
    Ignores black background by using a tissue mask if input_2d is provided.
    """
    if error_heatmap.max() == error_heatmap.min():
        return np.zeros_like(error_heatmap, dtype=np.uint8), 0.0
        
    tissue_mask = None
    if input_2d is not None:
        # Create a simple tissue mask (ignore dark background)
        tissue_mask = input_2d > 0.05
        
    try:
        if tissue_mask is not None and np.any(tissue_mask):
            # Compute Otsu ONLY on the tissue pixels to avoid separating Brain vs Background
            otsu_thresh = threshold_otsu(error_heatmap[tissue_mask])
        else:
            otsu_thresh = threshold_otsu(error_heatmap)
    except:
        otsu_thresh = 0.5
        
    final_thresh = threshold_override if threshold_override is not None else otsu_thresh
    binary_mask = (error_heatmap > final_thresh).astype(np.uint8)
    
    if tissue_mask is not None:
        # Mask out any spurious errors in the background
        binary_mask[~tissue_mask] = 0
        
    if postprocess:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        min_size = 50
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        new_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                new_mask[labels == i] = 1
        binary_mask = new_mask
        
    return binary_mask, final_thresh

def overlay_mask(image_2d, mask_2d, color=(255, 0, 0), alpha=0.5):
    """
    Overlays a binary mask onto a grayscale 2D image.
    image_2d: [0, 1] float
    mask_2d: {0, 1} uint8
    """
    img_rgb = cv2.cvtColor((image_2d * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    overlay = img_rgb.copy()
    overlay[mask_2d == 1] = color
    
    return cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

def encode_rle(mask):
    """
    Run-Length Encoding for COCO JSON format
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()
