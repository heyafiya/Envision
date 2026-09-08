import torch
import numpy as np
import cv2
import pandas as pd
from core.anomaly_segmentation import load_enhancement_model, compute_anomaly_map, apply_threshold
from core.train_enhancement import PreprocessedMRIDataset
from core.evaluate_segmentation import compute_segmentation_metrics
import json

def generate_synthetic_anomaly(image_2d):
    """
    Injects a synthetic tumor-like anomaly into a normal slice.
    Returns the anomalous image and the exact ground truth mask.
    """
    anomalous_img = image_2d.copy()
    gt_mask = np.zeros_like(image_2d, dtype=np.uint8)
    
    # Create a 20x20 circular anomaly near the center
    center = (112, 112)
    radius = 15
    cv2.circle(gt_mask, center, radius, 1, -1)
    
    # Blend the anomaly (make it brighter, like a T2 hyperintensity)
    anomaly_intensity = 0.8
    anomalous_img[gt_mask == 1] = anomalous_img[gt_mask == 1] * 0.3 + anomaly_intensity * 0.7
    
    # Add a bit of salt and pepper noise to simulate real-world noise triggering false positives
    noise = np.random.rand(*image_2d.shape)
    anomalous_img[noise > 0.99] = 1.0 # White noise dots
    
    return anomalous_img, gt_mask

def run_experiment():
    print("Loading normal data and Brain model...")
    dataset = PreprocessedMRIDataset('brain')
    model = load_enhancement_model('brain')
    
    metrics_before = []
    metrics_after = []
    
    # We will test on 10 random slices
    for i in range(min(10, len(dataset))):
        _, clean_tensor = dataset[i]
        clean_2d = clean_tensor[0].numpy()
        
        if np.sum(clean_2d) == 0:
            continue
            
        anomalous_img, gt_mask = generate_synthetic_anomaly(clean_2d)
        
        # 1. Compute Anomaly Map
        _, error_heatmap = compute_anomaly_map(model, anomalous_img)
        
        # 2. Before Post-processing
        mask_before, _ = apply_threshold(error_heatmap, threshold_override=None, input_2d=anomalous_img, postprocess=False)
        met_b = compute_segmentation_metrics(mask_before, gt_mask)
        metrics_before.append(met_b)
        
        # 3. After Post-processing
        mask_after, _ = apply_threshold(error_heatmap, threshold_override=None, input_2d=anomalous_img, postprocess=True)
        met_a = compute_segmentation_metrics(mask_after, gt_mask)
        metrics_after.append(met_a)
        
    # Average metrics
    avg_before = {k: np.mean([m[k] for m in metrics_before]) for k in metrics_before[0].keys()}
    avg_after = {k: np.mean([m[k] for m in metrics_after]) for k in metrics_after[0].keys()}
    
    results = []
    for k in avg_before.keys():
        results.append({
            "Metric": k,
            "Without Morphology (Raw)": round(avg_before[k], 4),
            "With Morphology (Clean)": round(avg_after[k], 4)
        })
        
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    df.to_csv("core/outputs/morphology_comparison.csv", index=False)
    
if __name__ == "__main__":
    run_experiment()
