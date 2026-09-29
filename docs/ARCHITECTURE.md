# Envision — System Architecture

## Overview

Envision is a self-supervised medical imaging pipeline designed
for anomaly detection and segmentation in brain and spine MRI scans.

The system is organized into five major stages, starting from
dataset discovery and preprocessing and progressing through
self-supervised learning, anomaly segmentation, and evaluation.

---

## End-to-End Pipeline

```text
MRI Dataset
     │
     ▼
┌─────────────────────────────┐
│ Stage 1: Data Discovery     │
│ • NIfTI file scanning       │
│ • Modality detection        │
│ • Dataset metadata          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Stage 2: Preprocessing      │
│ • NLM Denoising             │
│ • Percentile Clipping       │
│ • CLAHE Enhancement         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Stage 3: Self-Supervised    │
│ Learning                    │
│ • U-Net Autoencoder         │
│ • Healthy-scan training     │
│ • Rician-aware loss         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Stage 4: Anomaly            │
│ Segmentation                │
│ • Reconstruction error     │
│ • Otsu thresholding         │
│ • Morphological cleanup     │
│ • ROI mask generation       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Stage 5: Evaluation &       │
│ Reporting                   │
│ • Dice                      │
│ • IoU / Jaccard             │
│ • Sensitivity / Specificity │
│ • Hausdorff Distance        │
│ • PDF reporting             │
└─────────────────────────────┘
