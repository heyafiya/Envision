import nibabel as nib
import numpy as np

def load_nifti(path):
    """
    Load a NIfTI file (.nii or .nii.gz) using nibabel.
    Returns the loaded nibabel image.
    Works transparently on .nii and .nii.gz.
    """
    img = nib.load(path)
    return img

def get_middle_slice(volume, axis='axial'):
    """
    Get a normalized uint8 2D slice from a 3D volume for preview.
    Extracts the slice with the maximum sum (most visible anatomy) along the axis.
    """
    data = volume.get_fdata()
    
    if len(data.shape) == 2:
        slice_data = data
    elif len(data.shape) >= 3:
        # If it's effectively a 2D image saved as 3D (e.g. (240, 240, 1)), just squeeze it
        sq_data = np.squeeze(data)
        if len(sq_data.shape) == 2:
            slice_data = sq_data
        else:
            if axis == 'axial':
                ax_idx = 2
            elif axis == 'coronal':
                ax_idx = 1
            elif axis == 'sagittal':
                ax_idx = 0
            else:
                ax_idx = 2
                
            best_slice_idx = sq_data.shape[ax_idx] // 2
            max_sum = -1
            
            # Find the slice with the most signal
            for i in range(sq_data.shape[ax_idx]):
                if ax_idx == 2:
                    s = sq_data[:, :, i]
                elif ax_idx == 1:
                    s = sq_data[:, i, :]
                else:
                    s = sq_data[i, :, :]
                    
                s_sum = np.sum(s)
                if s_sum > max_sum:
                    max_sum = s_sum
                    best_slice_idx = i
                    
            if ax_idx == 2:
                slice_data = sq_data[:, :, best_slice_idx]
            elif ax_idx == 1:
                slice_data = sq_data[:, best_slice_idx, :]
            else:
                slice_data = sq_data[best_slice_idx, :, :]
    else:
        slice_data = np.zeros((10, 10))
        
    # Ensure slice_data is strictly 2D (in case of 4D volumes like (256, 256, z, c))
    while len(slice_data.shape) > 2:
        slice_data = slice_data[..., 0]
        
        
    # Normalize to 0-255 uint8 for preview
    min_val = np.min(slice_data)
    max_val = np.max(slice_data)
    if max_val - min_val == 0:
        return np.zeros_like(slice_data, dtype=np.uint8)
        
    normalized = (slice_data - min_val) / (max_val - min_val) * 255
    return normalized.astype(np.uint8)
