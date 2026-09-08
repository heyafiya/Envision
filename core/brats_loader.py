import os
import glob
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch

class BRATSDataset(Dataset):
    """
    Isolated loader for BRATS-2020 external data.
    Only loads T1/T2/FLAIR modalities from the specified external path.
    """
    def __init__(self, brats_dir="data/BRATS-2020", transform=None, limit=None):
        self.brats_dir = brats_dir
        self.transform = transform
        self.file_paths = []
        
        if os.path.exists(brats_dir):
            all_files = glob.glob(os.path.join(brats_dir, "**", "*.nii*"), recursive=True)
            # Filter to relevant modalities if needed
            self.file_paths = [f for f in all_files if any(m in f.lower() for m in ['t1', 't2', 'flair'])]
            
        if limit and limit < len(self.file_paths):
            self.file_paths = self.file_paths[:limit]
            
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = nib.load(path)
            data = img.get_fdata()
            # Basic processing: take middle slice
            if len(data.shape) >= 3:
                mid = data.shape[2] // 2
                slice_2d = data[:, :, mid]
                if len(slice_2d.shape) > 2:
                    slice_2d = slice_2d[:, :, 0]
            else:
                slice_2d = data
                
            # Resize to 224x224 for consistency
            import cv2
            slice_2d = cv2.resize(slice_2d, (224, 224))
            
            # Normalize to 0-1
            m, M = slice_2d.min(), slice_2d.max()
            if M > m:
                slice_2d = (slice_2d - m) / (M - m)
                
            tensor_data = torch.from_numpy(slice_2d).float()
            
            # Absolute foolproof shape enforcement
            if tensor_data.dim() == 3:
                tensor_data = tensor_data[:, :, 0]
            elif tensor_data.dim() > 3:
                tensor_data = tensor_data.reshape(224, 224)
                
            tensor_data = tensor_data.unsqueeze(0) # guaranteed [1, 224, 224]
            
            # Final safety check before yield
            if tensor_data.shape != (1, 224, 224):
                tensor_data = torch.zeros((1, 224, 224), dtype=torch.float32)
                
            return tensor_data, tensor_data # Autoencoder: input == target
        except Exception as e:
            # Return dummy on fail
            dummy = torch.zeros((1, 224, 224), dtype=torch.float32)
            return dummy, dummy

def get_brats_loader(batch_size=8, limit=None):
    dataset = BRATSDataset(limit=limit)
    if len(dataset) == 0:
        return None
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
