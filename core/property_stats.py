import numpy as np
import scipy.ndimage as ndimage

def compute_entropy(image):
    """Compute Shannon entropy of an 8-bit image."""
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 255))
    histogram = histogram[histogram > 0]
    probabilities = histogram / np.sum(histogram)
    return -np.sum(probabilities * np.log2(probabilities))

def compute_properties(volume_data, voxel_shape):
    """
    Compute image properties for a given volume data (numpy array).
    """
    # resolution/voxel shape
    resolution = "x".join(map(str, voxel_shape)) if voxel_shape is not None else "Unknown"
    
    # mean, std
    mean_val = np.mean(volume_data)
    std_val = np.std(volume_data)
    
    # contrast (std/mean or Michelson)
    contrast = std_val / mean_val if mean_val != 0 else 0
    
    # sharpness (Laplacian variance) & edge strength (Sobel)
    if len(volume_data.shape) >= 3:
        mid_slice = volume_data[:, :, volume_data.shape[2]//2]
    else:
        mid_slice = volume_data
        
    laplacian = ndimage.laplace(mid_slice)
    sharpness = np.var(laplacian)
    
    sobel_x = ndimage.sobel(mid_slice, axis=0)
    sobel_y = ndimage.sobel(mid_slice, axis=1)
    sobel_mag = np.hypot(sobel_x, sobel_y)
    edge_strength = np.mean(sobel_mag)
    
    # noise estimate (high-frequency energy -> std of laplacian)
    noise_est = np.std(laplacian)
    
    # complexity (entropy)
    min_val, max_val = np.min(mid_slice), np.max(mid_slice)
    if max_val - min_val > 0:
        norm_slice = ((mid_slice - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        complexity = compute_entropy(norm_slice)
    else:
        complexity = 0.0
        
    return {
        "resolution": resolution,
        "mean": float(mean_val),
        "std": float(std_val),
        "contrast": float(contrast),
        "sharpness": float(sharpness),
        "edge_strength": float(edge_strength),
        "noise_estimate": float(noise_est),
        "complexity": float(complexity)
    }
