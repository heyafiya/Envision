import os
import json
import pandas as pd
import nibabel as nib
from core.data_discovery import build_manifest
from core.preprocessing import preprocess_pipeline
from core.property_stats import compute_properties

print("Loading manifest...")
manifest_path = "core/manifest.csv"
if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) <= 2:
    df = build_manifest()
else:
    df = pd.read_csv(manifest_path)

df_rep = df[df['representative'] == True] if 'representative' in df.columns else df
total_files = len(df_rep)

all_stats = {}
os.makedirs("core/outputs", exist_ok=True)

print(f"Starting batch preprocessing on {total_files} files...")
for idx, row in df_rep.reset_index().iterrows():
    print(f"Processing {idx+1}/{total_files}: {row['filepath']}")
    try:
        img = nib.load(row['filepath'])
        raw_data = img.get_fdata()
        preprocessed_data = preprocess_pipeline(raw_data)
        
        # Save NIfTI
        out_dir = os.path.join("core/outputs/preprocessed", row['region'], row['status'])
        os.makedirs(out_dir, exist_ok=True)
        
        filename = os.path.basename(row['filepath'])
        out_path = os.path.join(out_dir, filename)
        
        prep_img = nib.Nifti1Image(preprocessed_data, img.affine, img.header)
        nib.save(prep_img, out_path)
        
        # Compute Stats
        stats = compute_properties(preprocessed_data, img.header.get_zooms())
        all_stats[out_path] = {
            "patient_id": row['patient_id'],
            "region": row['region'],
            "modality": row['modality'],
            "stats": stats
        }
    except Exception as e:
        print(f"Error on {row['filepath']}: {e}")
        all_stats[row['filepath']] = {"error": str(e)}

with open("core/outputs/stage2_property_stats.json", "w") as f:
    json.dump(all_stats, f, indent=4)

print("Batch preprocessing complete!")
