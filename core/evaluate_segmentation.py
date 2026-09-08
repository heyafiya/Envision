import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import morphology

def extract_surface(mask):
    """
    Extracts surface boundary of a binary mask.
    """
    eroded = morphology.binary_erosion(mask, iterations=1)
    return mask ^ eroded

def compute_segmentation_metrics(pred_mask, ref_mask):
    """
    pred_mask: numpy array {0, 1}
    ref_mask: numpy array {0, 1} - approximate reference
    """
    metrics = {}
    
    tp = np.sum((pred_mask == 1) & (ref_mask == 1))
    tn = np.sum((pred_mask == 0) & (ref_mask == 0))
    fp = np.sum((pred_mask == 1) & (ref_mask == 0))
    fn = np.sum((pred_mask == 0) & (ref_mask == 1))
    
    # 1. Dice & F1 (Mathematically equivalent)
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    metrics['Dice'] = float(dice)
    metrics['F1 Score'] = float(dice)
    
    # 2. Jaccard (IoU)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    metrics['Jaccard (IoU)'] = float(iou)
    
    # 3. Accuracy
    total = tp + tn + fp + fn
    metrics['Accuracy'] = float((tp + tn) / total) if total > 0 else 1.0
    
    # 4. Sensitivity (Recall)
    metrics['Sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    
    # 5. Specificity
    metrics['Specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    
    # 6. Precision
    metrics['Precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    
    # 7. Relative Volume Error (RVE)
    vol_pred = np.sum(pred_mask)
    vol_ref = np.sum(ref_mask)
    metrics['RVE (%)'] = float(abs(vol_pred - vol_ref) / vol_ref * 100) if vol_ref > 0 else 0.0
    
    # Distance metrics
    pred_pts = np.argwhere(extract_surface(pred_mask))
    ref_pts = np.argwhere(extract_surface(ref_mask))
    
    if len(pred_pts) > 0 and len(ref_pts) > 0:
        # 8. Hausdorff Distance
        hd1 = directed_hausdorff(pred_pts, ref_pts)[0]
        hd2 = directed_hausdorff(ref_pts, pred_pts)[0]
        metrics['Hausdorff Distance'] = float(max(hd1, hd2))
        
        # 9. Average Surface Distance (ASSD)
        from scipy.spatial import cKDTree
        tree_ref = cKDTree(ref_pts)
        tree_pred = cKDTree(pred_pts)
        
        d_pred_to_ref, _ = tree_ref.query(pred_pts)
        d_ref_to_pred, _ = tree_pred.query(ref_pts)
        
        assd = (np.sum(d_pred_to_ref) + np.sum(d_ref_to_pred)) / (len(pred_pts) + len(ref_pts))
        metrics['Avg Surface Distance'] = float(assd)
    else:
        metrics['Hausdorff Distance'] = 0.0
        metrics['Avg Surface Distance'] = 0.0

    return {k: round(v, 4) for k, v in metrics.items()}
