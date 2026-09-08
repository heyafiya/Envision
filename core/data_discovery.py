import os
import glob
import pandas as pd
from pathlib import Path

def discover_files(base_dir="data"):
    """
    Recursively glob data for *.nii and *.nii.gz files.
    """
    base_path = Path(base_dir)
    nii_files = list(base_path.rglob("*.nii")) + list(base_path.rglob("*.nii.gz"))
    
    records = []
    
    for filepath in nii_files:
        path_str = str(filepath).replace('\\', '/')
        parts = path_str.lower().split('/')
        
        # Determine region and status based on path
        if any('brain' in p for p in parts):
            region = 'brain'
        elif any('spine' in p for p in parts):
            region = 'spine'
        else:
            # Try to guess based on filename if not in path
            if 'brp' in filepath.name.lower() or 's' in filepath.name.lower():
                region = 'brain'
            elif 'sp' in filepath.name.lower():
                region = 'spine'
            else:
                continue
            
        if any('normal' in p for p in parts):
            status = 'normal'
        elif any('pathological' in p for p in parts):
            status = 'pathological'
        else:
            status = 'unknown'
            
        # Determine patient ID - look for S, BRP, SP followed by number in parts
        patient_id = "Unknown"
        # We check the original case parts
        orig_parts = str(filepath).replace('\\', '/').split('/')
        for part in orig_parts:
            if part.upper().startswith(('S', 'BRP', 'SP')) and any(c.isdigit() for c in part):
                # Try to extract exact patient ID if it's like S1, BRP1, SP1
                patient_id = part
                break
                
        # Dimensionality
        dims = "3D"  # Default
        if any('2d' in p for p in parts):
            dims = "2D"
        elif any('3d' in p for p in parts):
            dims = "3D"
        else:
            if region == 'spine':
                dims = "2D"
                
        # Modality inference
        modality = "Unknown"
        filename = filepath.name.lower()
        if region == 'brain':
            if 'flair' in filename:
                modality = "FLAIR"
            elif 't1ce' in filename or 't1c' in filename:
                modality = "T1CE"
            elif 't1' in filename and not ('t1ce' in filename or 't1c' in filename):
                modality = "T1"
            elif 't2' in filename:
                modality = "T2"
        elif region == 'spine':
            if 'stir' in filename:
                modality = "STIR"
            elif 't1c' in filename or 't1ce' in filename:
                modality = "T1C"
            elif 't1' in filename:
                modality = "T1"
            elif 't2' in filename:
                modality = "T2"
                
        records.append({
            "patient_id": patient_id,
            "region": region,
            "status": status,
            "modality": modality,
            "dims": dims,
            "filepath": str(filepath)
        })
        
    df = pd.DataFrame(records)
    
    # Representative selection
    if not df.empty:
        df['representative'] = True
        
        # Group by patient_id, region, status, modality
        for (pid, reg, stat, mod), group in df.groupby(['patient_id', 'region', 'status', 'modality']):
            if len(group) > 1:
                df.loc[group.index, 'representative'] = False
                
                # Filter by 2D preference
                dims_2d = group[group['dims'] == '2D']
                if not dims_2d.empty:
                    candidates = dims_2d
                else:
                    candidates = group
                    
                # Pref CLEAR/HST
                clear_hst = candidates[candidates['filepath'].str.lower().str.contains('clear|hst')]
                if not clear_hst.empty:
                    selected_idx = clear_hst.sort_values('filepath').index[0]
                else:
                    selected_idx = candidates.sort_values('filepath').index[0]
                    
                df.loc[selected_idx, 'representative'] = True
    
    return df

def build_manifest():
    """
    Build the manifest and save it to core/manifest.csv.
    """
    os.makedirs("core", exist_ok=True)
    df = discover_files()
    df.to_csv("core/manifest.csv", index=False)
    return df

if __name__ == "__main__":
    df = build_manifest()
    print(f"Discovered {len(df)} files.")
