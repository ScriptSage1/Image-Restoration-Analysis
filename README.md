# Image Distortion Analysis and Restoration

## Overview

This project aims to **analyze the effects of various image distortions** and compare the performance of **traditional algorithms versus deep learning methods** for restoring degraded images using **quantitative metrics (PSNR, SSIM)** and **qualitative visual comparisons**.

---

## Project Structure

```
Image-Distortion-Analysis/
│
├── data/
│   ├── original/             # Place your original PNG/JPG images here
│   │   ├── 0060.png
│   │   ├── 0267.png
│   │   └── 0268.png
│   └── distorted/            # Generated distorted images are saved here
│       └── gaussian_noise/
│           └── ...
│
├── distortions/              # Modules for applying different distortions
│   ├── __init__.py
│   ├── color_shift.py
│   ├── down_sampling.py
│   ├── gaussian_blur.py
│   ├── gaussian_noise.py
│   └── jpeg_compression.py
│
├── restorations/             # Modules for image restoration algorithms
│   ├── __init__.py
│   ├── deep_learning/
│   │   ├── __init__.py
│   │   ├── deep_learning_denoiser.py   # Applies the DRUNet model
│   │   └── drunet_model.py             # DRUNet architecture definition
│   └── traditional/
│       ├── __init__.py
│       └── Gaussian_Noise/
│           ├── __init__.py
│           ├── median_filter.py
│           └── non_local_means.py
│
├── results/                  # Output directory for analysis results
│   └── for_GaussianNoise/
│       ├── closeup_... .png
│       ├── full_comparison_... .png
│       ├── metrics_chart_... .png
│       └── metrics_... .csv
│
│   
│
├── .gitignore
├── analysis.ipynb            # Interactive notebook for exploration
├── calculate_metrics.py      # Main script to run the full analysis pipeline
└── requirements.txt          # Python dependencies
```

---
