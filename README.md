# Envision

**A Self-Supervised U-Net Architecture for Multi-Modal MRI Anomaly Segmentation**

Envision is an advanced Medical AI pipeline built to detect anomalies (tumors, lesions, degenerative discs) in multi-modal MRI scans (Brain and Spine). It completely bypasses the need for expensive, hand-drawn clinical ground-truth masks by utilizing a **Self-Supervised Learning** paradigm. The AI is trained exclusively on healthy scans to learn a mathematical representation of normative human anatomy. Anomalies are then dynamically detected as reconstruction errors.

---

## 🚀 Features & Pipeline Stages

### Stage 1: Automated Data Discovery & Analysis
*   Recursively scans the local `data/` directory.
*   Automatically builds a manifest of all available NIfTI files (`.nii`, `.nii.gz`), categorizing them by region (Brain/Spine), status (Normal/Pathological), and modality (T1, T2, FLAIR, T1CE, STIR).
*   Extracts dynamic structural metadata (dimensions, voxel spacing).

### Stage 2: Physics-Informed Preprocessing
*   **Non-Local Means (NLM) Denoising**: Tailored to reduce Rician noise (the physical noise profile of MRI equipment) without destroying high-frequency tissue boundaries like Gaussian blur does.
*   **Percentile Clipping**: Removes extreme hyper-intense RF-coil artifacts.
*   **CLAHE Enhancement**: Standardizes inter-scanner contrast.

### Stage 3: Self-Supervised Deep Learning
*   **Architecture**: Deep U-Net Autoencoder with Skip Connections to preserve high-resolution spatial details during the encoding/decoding bottleneck.
*   **Loss Function**: Custom **Rician-Aware Negative Log-Likelihood (NLL) Loss** specifically modeled for MRI physics, eliminating the blurriness caused by standard MSE loss.
*   **Training**: Trained *only* on the Normal dataset to establish a baseline of healthy anatomy.

### Stage 4: Live Anomaly Segmentation (Inference)
*   **Reconstruction Error Map**: Computes the pixel-wise difference between the input scan and the model's "healthy" reconstruction attempt, generating an Attention/Grad-CAM style heatmap.
*   **Adaptive Otsu's Thresholding**: Dynamically computes the mathematically optimal binarization cutoff based on the unique noise floor of the uploaded patient's scan.
*   **Morphological Cleanup**: Applies mathematical Opening/Closing algorithms to remove 1-pixel noise artifacts and isolate contiguous Regions of Interest (ROI).
*   **COCO Export**: Generates and exports RLE-encoded binary masks.

### Stage 5: Technical Benchmarking & Reporting
*   Calculates standard clinical evaluation metrics: **Dice Coefficient, Jaccard Index (IoU), Sensitivity, Specificity, and Hausdorff Distance**.
*   Directly benchmarks against state-of-the-art literature baselines (e.g., BraTS 2020 solutions).
*   Native PDF Generation for technical reports outlining clinical translation limitations.

---

## ⚙️ How to Run

### 1. Requirements
Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```
*(Required packages include `streamlit`, `torch`, `torchvision`, `numpy`, `pandas`, `nibabel`, `opencv-python-headless`, `fpdf2`, `scikit-image`)*

### 2. Dataset Structure
Place your raw MRI `.nii` or `.nii.gz` files in the following directory structure inside the root folder:

```text
data/
├── brain/
│   ├── normal/
│   └── pathological/
└── spine/
    ├── normal/
    └── pathological/
```

### 3. Launching the Application
The entire pipeline is controlled via an interactive Streamlit dashboard.

```bash
python -m streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser.

---

## 📂 Project Structure

```text
Envision/
├── app.py                      # Streamlit application entry point
├── core/
│   ├── anomaly_segmentation.py # Stage 4 Inference & Mask Generation
│   ├── data_discovery.py       # Stage 1 Dataset Manifest Builder
│   ├── evaluate_segmentation.py# Stage 5 Metrics Calculation
│   ├── io_utils.py             # NIfTI file loading and slice extraction
│   ├── preprocessing.py        # Stage 2 NLM and CLAHE logic
│   ├── property_stats.py       # Statistical analysis of MRI slices
│   ├── train_enhancement.py    # Training logic & Rician Loss function
│   ├── models/                 # Model definitions
│   │   └── autoencoder.py      # U-Net Architecture
│   └── outputs/                # Generated masks, PDFs, and metrics
├── pages/                      
│   ├── 1_dataset_analysis.py   # UI: Dataset Exploration
│   ├── 2_preprocessing.py      # UI: Denoising & Enhancement
│   ├── 3_enhancement_model.py  # UI: Model Training & Evaluation
│   ├── 4_roi_segmentation.py   # UI: Live Demo Inference
│   └── 5_technical_report.py   # UI: Paper Benchmarking & PDF Export
└── README.md                   # Project documentation
```

---

## 🏥 Clinical Limitations
This tool is a hackathon prototype demonstrating theoretical proof-of-concept for self-supervised anomaly detection. It is **not** cleared for medical use. Deployment requires a significantly expanded normative dataset, native DICOM PACS integration, and a radiologist-in-the-loop manual threshold verification interface.
