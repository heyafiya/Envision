import numpy as np
import cv2
from skimage.exposure import equalize_adapthist

def basic_intensity_normalization(data):
    """
    Basic artifact correction: clips extreme values (e.g. top and bottom 1%) 
    to remove hyper-intense artifacts, and normalizes.
    """
    p1, p99 = np.percentile(data, (1, 99))
    clipped = np.clip(data, p1, p99)
    return clipped

def rescale_intensities(data):
    """
    Rescales intensities to [0, 1].
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data, dtype=np.float32)
    return (data - min_val) / (max_val - min_val)

def denoise_nlm(data):
    """
    Denoise using Non-Local Means.
    MRI noise is Rician; NLM is highly effective at preserving edges 
    unlike generic Gaussian blur which blurs tissue boundaries.
    """
    is_3d = len(data.shape) >= 3
    
    # Scale to 0-255 uint8 for cv2
    scaled = np.clip(data * 255, 0, 255).astype(np.uint8)
    
    if is_3d:
        denoised = np.zeros_like(scaled)
        for i in range(scaled.shape[2]): # assuming axial/depth is dim 2
            slice_2d = scaled[:, :, i]
            # h=3 for mild denoising
            d_slice = cv2.fastNlMeansDenoising(slice_2d, None, h=3, templateWindowSize=7, searchWindowSize=21)
            denoised[:, :, i] = d_slice
        return denoised.astype(np.float32) / 255.0
    else:
        d_img = cv2.fastNlMeansDenoising(scaled, None, h=3, templateWindowSize=7, searchWindowSize=21)
        return d_img.astype(np.float32) / 255.0

def baseline_enhancement(data, method='CLAHE'):
    """
    Baseline enhancement using CLAHE.
    """
    if method == 'CLAHE':
        # CLAHE operates slice-by-slice
        is_3d = len(data.shape) >= 3
        if is_3d:
            enhanced = np.zeros_like(data)
            for i in range(data.shape[2]):
                enhanced[:, :, i] = equalize_adapthist(data[:, :, i], clip_limit=0.01)
            return enhanced
        else:
            return equalize_adapthist(data, clip_limit=0.01)
    return data

def preprocess_pipeline(data):
    # 1. Bias-field/intensity norm
    d1 = basic_intensity_normalization(data)
    # 2. Rescale to [0,1]
    d2 = rescale_intensities(d1)
    # 3. Denoise (NLM)
    d3 = denoise_nlm(d2)
    # 4. CLAHE
    d4 = baseline_enhancement(d3, method='CLAHE')
    
    return d4
