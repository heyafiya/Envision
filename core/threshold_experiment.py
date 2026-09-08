import torch
import numpy as np
import cv2
import pandas as pd
from core.anomaly_segmentation import load_enhancement_model, compute_anomaly_map, apply_threshold
from core.train_enhancement import PreprocessedMRIDataset
from core.evaluate_segmentation import compute_segmentation_metrics

def generate_synthetic_anomaly(image_2d):
    anomalous_img = image_2d.copy()
    gt_mask = np.zeros_like(image_2d, dtype=np.uint8)
    
    center = (112, 112)
    radius = 15
    cv2.circle(gt_mask, center, radius, 1, -1)
    
    anomaly_intensity = 0.8
    anomalous_img[gt_mask == 1] = anomalous_img[gt_mask == 1] * 0.3 + anomaly_intensity * 0.7
    
    noise = np.random.rand(*image_2d.shape)
    anomalous_img[noise > 0.99] = 1.0 
    
    return anomalous_img, gt_mask

def run_experiment():
    print("Loading normal data and Brain model...")
    dataset = PreprocessedMRIDataset('brain')
    model = load_enhancement_model('brain')
    
    metrics_fixed = []
    metrics_adaptive = []
    
    # Pre-compute a global fixed threshold based on average Otsu over the first few, 
    # or just pick a common global error threshold like 0.25 (since heatmap is normalized 0 to 1).
    FIXED_THRESHOLD = 0.25
    
    for i in range(min(15, len(dataset))):
        _, clean_tensor = dataset[i]
        clean_2d = clean_tensor[0].numpy()
        
        if np.sum(clean_2d) == 0:
            continue
            
        anomalous_img, gt_mask = generate_synthetic_anomaly(clean_2d)
        _, error_heatmap = compute_anomaly_map(model, anomalous_img)
        
        # 1. Fixed Global Threshold
        mask_fixed, _ = apply_threshold(error_heatmap, threshold_override=FIXED_THRESHOLD, input_2d=anomalous_img, postprocess=True)
        met_fixed = compute_segmentation_metrics(mask_fixed, gt_mask)
        metrics_fixed.append(met_fixed)
        
        # 2. Adaptive Per-Patient Otsu
        mask_adaptive, _ = apply_threshold(error_heatmap, threshold_override=None, input_2d=anomalous_img, postprocess=True)
        met_adaptive = compute_segmentation_metrics(mask_adaptive, gt_mask)
        metrics_adaptive.append(met_adaptive)
        
    avg_fixed = {k: np.mean([m[k] for m in metrics_fixed]) for k in metrics_fixed[0].keys()}
    avg_adaptive = {k: np.mean([m[k] for m in metrics_adaptive]) for k in metrics_adaptive[0].keys()}
    
    results = []
    for k in avg_fixed.keys():
        results.append({
            "Metric": k,
            "Fixed Global Threshold (0.25)": round(avg_fixed[k], 4),
            "Adaptive Per-Patient (Otsu)": round(avg_adaptive[k], 4)
        })
        
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
if __name__ == "__main__":
    run_experiment()
