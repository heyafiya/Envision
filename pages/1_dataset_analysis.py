import streamlit as st
import pandas as pd
import os
import json
from core.data_discovery import build_manifest
from core.io_utils import load_nifti, get_middle_slice
from core.property_stats import compute_properties

st.set_page_config(page_title="Stage 1: Dataset Analysis", layout="wide")

st.title("Stage 1: Dataset Exploration, Analysis, and Preparation")

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
    st.warning("No dataset files found. Please ensure data is present in 'data/brain' and 'data/spine'.")
    st.stop()

df_rep = df[df['representative'] == True] if 'representative' in df.columns else df

# 2. Config Dict for Markdown
CHALLENGES = {
    "Brain": {
        "T1": "Rician noise is prominent in background regions. High variance in slice thickness across acquisitions.",
        "T2": "Intensity inhomogeneity due to bias field. Uneven contrast across different patients.",
        "FLAIR": "Fluid attenuation is inconsistent in pathological cases. Edema boundaries can be blurry.",
        "T1CE": "Contrast enhancement varies based on dosage and timing, making intensity standardization difficult."
    },
    "Spine": {
        "T1": "Vertebral body intensity varies significantly. Rician noise is present.",
        "T2": "CSF intensity saturation can cause artifacts. High variance in slice thickness.",
        "STIR": "Fat suppression is sometimes incomplete. Lower SNR compared to T1/T2."
    }
}

# 3. Train/Test Split Table
st.header("Dataset Split Strategy")
st.markdown("We are following the official Hackathon constraints:")
split_data = [
    {"Dataset": "Brain", "Status": "Normal", "Usage": "Training (Brain)"},
    {"Dataset": "Spine", "Status": "Normal", "Usage": "Training (Spine)"},
    {"Dataset": "Brain", "Status": "Pathological", "Usage": "Test-Only"},
    {"Dataset": "Spine", "Status": "Pathological", "Usage": "Test-Only"},
]
st.table(pd.DataFrame(split_data))

# 4. Modality Comparison
st.header("Modality Counts")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Brain Submodalities")
    brain_df = df_rep[df_rep['region'] == 'brain']
    if not brain_df.empty:
        st.dataframe(brain_df.groupby(['modality', 'status']).size().unstack(fill_value=0))
    else:
        st.write("No brain data found.")
        
with col2:
    st.subheader("Spine Submodalities")
    spine_df = df_rep[df_rep['region'] == 'spine']
    if not spine_df.empty:
        st.dataframe(spine_df.groupby(['modality', 'status']).size().unstack(fill_value=0))
    else:
        st.write("No spine data found.")

# 5. Dataset Challenges
st.header("Dataset Challenges")
tab_brain, tab_spine = st.tabs(["Brain", "Spine"])
with tab_brain:
    for mod, desc in CHALLENGES["Brain"].items():
        st.markdown(f"**{mod}**: {desc}")
with tab_spine:
    for mod, desc in CHALLENGES["Spine"].items():
        st.markdown(f"**{mod}**: {desc}")

# 6. Patient Dropdown and Previews
st.header("Data Preview")

col_region, col_patient = st.columns(2)
with col_region:
    regions = sorted(df_rep['region'].unique().tolist())
    selected_region = st.selectbox("Select Region", regions)

with col_patient:
    region_patients = df_rep[df_rep['region'] == selected_region]
    patients = sorted(region_patients['patient_id'].unique().tolist())
    selected_patient = st.selectbox("Select Patient", patients)

if selected_patient:
    patient_files = df_rep[(df_rep['patient_id'] == selected_patient) & (df_rep['region'] == selected_region)]
    st.write(f"Region: {patient_files['region'].iloc[0].capitalize()} | Status: {patient_files['status'].iloc[0].capitalize()}")
    
    modalities = patient_files['modality'].tolist()
    cols = st.columns(max(len(modalities), 1))
    
    for idx, row in patient_files.reset_index().iterrows():
        try:
            img = load_nifti(row['filepath'])
            
            slice_2d = get_middle_slice(img)
            
            # Failsafe: if the image still has more than 2 dimensions (e.g. module caching issue), force it to 2D
            if len(slice_2d.shape) > 2:
                slice_2d = slice_2d[..., 0]
                
            with cols[idx]:
                st.image(slice_2d, caption=f"{row['modality']} ({row['dims']})", use_container_width=True)
        except Exception as e:
            with cols[idx]:
                st.error(f"Error loading {row['modality']}: {e}")

    # Display properties for the currently selected patient on screen
    st.subheader(f"Computed Properties for {selected_patient}")
    stats_records = []
    for idx, row in patient_files.reset_index().iterrows():
        try:
            img = load_nifti(row['filepath'])
            data = img.get_fdata()
            stats = compute_properties(data, img.header.get_zooms())
            stats['Modality'] = row['modality']
            stats_records.append(stats)
        except Exception:
            pass
    if stats_records:
        st.dataframe(pd.DataFrame(stats_records).set_index('Modality'))

# 7. Compute & Export Stats (Button)
st.header("Batch Compute & Export")
st.markdown("Run this to compute properties for **ALL 129 representative files** (Brain and Spine) and save them to `core/outputs/stage1_dataset_stats.json`.")
if st.button("Run Batch Property Assessment"):
    with st.spinner("Computing statistics for all representative volumes..."):
        all_stats = {}
        os.makedirs("core/outputs", exist_ok=True)
        
        progress_bar = st.progress(0)
        total_files = len(df_rep)
        
        for idx, row in df_rep.reset_index().iterrows():
            try:
                img = load_nifti(row['filepath'])
                data = img.get_fdata()
                stats = compute_properties(data, img.header.get_zooms())
                
                all_stats[row['filepath']] = {
                    "patient_id": row['patient_id'],
                    "region": row['region'],
                    "modality": row['modality'],
                    "stats": stats
                }
            except Exception as e:
                all_stats[row['filepath']] = {"error": str(e)}
                
            progress_bar.progress((idx + 1) / total_files)
            
        with open("core/outputs/stage1_dataset_stats.json", "w") as f:
            json.dump(all_stats, f, indent=4)
            
        st.success("Successfully computed and saved to core/outputs/stage1_dataset_stats.json!")
