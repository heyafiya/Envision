import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import nibabel as nib
from core.data_discovery import build_manifest
from core.io_utils import load_nifti, get_middle_slice
from core.property_stats import compute_properties
from core.preprocessing import preprocess_pipeline

st.set_page_config(page_title="Stage 2: Preprocessing", layout="wide")
st.title("Stage 2: Preprocessing & Enhancement")

# 1. Build / Load Manifest
@st.cache_data
def load_dataset_manifest_v3():
    manifest_path = "core/manifest.csv"
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) <= 2:
        df = build_manifest()
    else:
        try:
            df = pd.read_csv(manifest_path)
        except pd.errors.EmptyDataError:
            df = build_manifest()
    return df

with st.spinner("Loading dataset manifest..."):
    df = load_dataset_manifest_v3()

if df.empty:
    st.warning("No dataset files found.")
    st.stop()

df_rep = df[df['representative'] == True] if 'representative' in df.columns else df

# 2. Patient Dropdown and Previews
st.header("Preprocessing Preview")

col_region, col_patient = st.columns(2)
with col_region:
    regions = sorted(df_rep['region'].unique().tolist())
    selected_region = st.selectbox("Select Region", regions, key='prep_reg')

with col_patient:
    region_patients = df_rep[df_rep['region'] == selected_region]
    patients = sorted(region_patients['patient_id'].unique().tolist())
    selected_patient = st.selectbox("Select Patient", patients, key='prep_pat')

if selected_patient:
    patient_files = df_rep[(df_rep['patient_id'] == selected_patient) & (df_rep['region'] == selected_region)]
    modalities = patient_files['modality'].tolist()
    selected_modality = st.selectbox("Select Modality", modalities)
    
    if selected_modality:
        row = patient_files[patient_files['modality'] == selected_modality].iloc[0]
        
        try:
            with st.spinner("Applying preprocessing pipeline..."):
                img = load_nifti(row['filepath'])
                raw_data = img.get_fdata()
                
                # Extract a single 2D slice first for fast preview
                if len(raw_data.shape) > 3:
                    raw_data = raw_data[:, :, :, 0]
                if len(raw_data.shape) >= 3:
                    mid = raw_data.shape[2] // 2
                    raw_slice_float = raw_data[:, :, mid]
                else:
                    raw_slice_float = raw_data
                    
                # Preprocess ONLY the 2D slice
                prep_slice_float = preprocess_pipeline(raw_slice_float)
                
                # Normalize for display
                def to_uint8(data):
                    m, M = data.min(), data.max()
                    if M > m:
                        return ((data - m) / (M - m) * 255).astype(np.uint8)
                    return np.zeros_like(data, dtype=np.uint8)
                    
                slice_raw = to_uint8(raw_slice_float)
                slice_prep = to_uint8(prep_slice_float)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(slice_raw, caption=f"RAW: {row['modality']} ({row['dims']})", use_container_width=True)
                with col2:
                    st.image(slice_prep, caption=f"PREPROCESSED: {row['modality']}", use_container_width=True)
                    
                # Compute stats on the 2D slices for the preview
                # Fake zoom for 2D
                zooms = img.header.get_zooms()[:2] if len(img.header.get_zooms()) >= 2 else (1.0, 1.0)
                raw_stats = compute_properties(raw_slice_float, zooms)
                prep_stats = compute_properties(prep_slice_float, zooms)
                
                stats_df = pd.DataFrame([raw_stats, prep_stats], index=["Raw (Slice)", "Preprocessed (Slice)"]).T
                st.subheader("Property Comparison")
                st.dataframe(stats_df)
                
        except Exception as e:
            st.error(f"Error processing: {e}")

# 3. Batch Preprocessing
st.header("Batch Preprocessing")
st.markdown("""
Run this to apply the Denoising (NLM) + Rescaling + Bias Correction + CLAHE Enhancement pipeline 
to **ALL** files in the manifest. The outputs will be saved to `core/outputs/preprocessed/`.
""")

if st.button("Run on entire dataset"):
    progress_bar = st.progress(0)
    total_files = len(df_rep)
    
    all_stats = {}
    os.makedirs("core/outputs", exist_ok=True)
    
    for idx, row in df_rep.reset_index().iterrows():
        try:
            img = load_nifti(row['filepath'])
            raw_data = img.get_fdata()
            preprocessed_data = preprocess_pipeline(raw_data)
            
            # Save NIfTI
            out_dir = os.path.join("core/outputs/preprocessed", row['region'], row['status'])
            os.makedirs(out_dir, exist_ok=True)
            
            filename = os.path.basename(row['filepath'])
            out_path = os.path.join(out_dir, filename)
            
            # Omit old header so nibabel generates a clean one with the float32 datatype
            prep_img = nib.Nifti1Image(preprocessed_data.astype(np.float32), img.affine)
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
            all_stats[row['filepath']] = {"error": str(e)}
            
        progress_bar.progress((idx + 1) / total_files)
        
    # Save stats
    with open("core/outputs/stage2_property_stats.json", "w") as f:
        json.dump(all_stats, f, indent=4)
        
    # Save justification
    justification = """# Stage 2 Preprocessing Justification\n\n## Denoising: Non-Local Means (NLM) vs Gaussian\nMagnetic Resonance Imaging (MRI) is predominantly affected by **Rician noise** rather than generic Gaussian noise. Rician noise depends on the signal intensity, meaning background regions (air) have different noise characteristics than dense tissue. Applying a generic Gaussian blur would destructively smooth out high-frequency tissue boundaries and fine structural details (like edema margins or nerve roots). \n\nWe employ **Non-Local Means (NLM)** because it compares patches of the image to find similar structures before averaging, effectively reducing Rician noise while strictly preserving the sharp edges necessary for accurate Region of Interest (ROI) segmentation in Stage 4.\n\n## Artifact Correction: Percentile Clipping & Normalization\nIntensities in MRI are not absolute (unlike CT Hounsfield units). A T1-weighted scan for Patient A might range from 0-500, while Patient B's ranges from 0-2500. Furthermore, RF-coil bias fields cause hyper-intense regions. We clip the extreme 1% and 99% intensity percentiles to remove outlier artifacts (like fat hyper-intensities or skull boundary artifacts) and min-max scale the volume to a standardized `[0, 1]` range.\n\n## Enhancement: Contrast-Limited Adaptive Histogram Equalization (CLAHE)\nStandard Histogram Equalization (HE) often over-amplifies noise in homogeneous regions (like the background). CLAHE divides the MRI slice into contextual tiles and limits the contrast amplification, providing a robust baseline enhancement. This serves as the **Reference Point** that our deep learning model must beat in Stage 3.\n"""
    with open("core/outputs/stage2_preprocessing_justification.md", "w") as f:
        f.write(justification)
        
    st.success("Batch preprocessing complete! Files saved to `core/outputs/preprocessed/`")
