import os
import pandas as pd
import numpy as np
import cv2
import nibabel as nib
from core.io_utils import get_middle_slice
from core.preprocessing import preprocess_pipeline
from core.anomaly_segmentation import load_enhancement_model, compute_anomaly_map, apply_threshold

def run_audit():
    manifest_path = "core/manifest.csv"
    if not os.path.exists(manifest_path):
        print("Manifest not found.")
        return
        
    df = pd.read_csv(manifest_path)
    # Filter only pathological patients
    df_path = df[df['status'] == 'pathological']
    
    print(f"Total pathological entries: {len(df_path)}")
    
    # Load models
    brain_model = load_enhancement_model('Brain')
    spine_model = load_enhancement_model('Spine')
    
    failed_patients = []
    
    for idx, row in df_path.iterrows():
        region = row['region']
        patient_id = row['patient_id']
        modality = row['modality']
        file_path = row['filepath']
        
        try:
            img = nib.load(file_path)
            raw_slice = get_middle_slice(img)
            if len(raw_slice.shape) > 2:
                raw_slice = raw_slice[:, :, 0]
                
            raw_slice = cv2.resize(raw_slice, (224, 224))
            
            prep_slice = preprocess_pipeline(raw_slice)
            prep_slice = cv2.resize(prep_slice, (224, 224))
            
            p_min, p_max = prep_slice.min(), prep_slice.max()
            if p_max > p_min:
                prep_slice_norm = (prep_slice - p_min) / (p_max - p_min)
            else:
                prep_slice_norm = prep_slice
                
            model = brain_model if region == 'Brain' else spine_model
            _, error_heatmap = compute_anomaly_map(model, prep_slice_norm)
            
            mask, _ = apply_threshold(error_heatmap, None, prep_slice_norm, postprocess=True)
            
            # Check mask area
            total_pixels = mask.size
            active_pixels = np.sum(mask)
            ratio = active_pixels / total_pixels
            
            if ratio == 0:
                failed_patients.append(f"[{region}] {patient_id} ({modality}): EMPTY MASK (0%)")
            elif ratio > 0.4:  # If anomaly is more than 40% of the slice, it's likely a massive false positive blob
                failed_patients.append(f"[{region}] {patient_id} ({modality}): MASSIVE MASK ({ratio*100:.1f}%)")
                
        except Exception as e:
            failed_patients.append(f"[{region}] {patient_id} ({modality}): ERROR ({str(e)})")

    print("\n--- FINAL AUDIT RESULTS ---")
    if len(failed_patients) == 0:
        print("All pathological patients passed segmentation without catastrophic failure.")
    else:
        print(f"Warning: {len(failed_patients)} patients flagged:")
        for fp in failed_patients:
            print(" - " + fp)

if __name__ == "__main__":
    run_audit()
