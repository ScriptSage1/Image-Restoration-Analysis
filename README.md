Image Distortion Analysis and Restoration

Overview

This project aims to analyze the effects of various image distortions and compare the performance of traditional algorithms versus deep learning methods for restoring degraded images. Currently, it focuses on applying Gaussian noise and evaluating restoration techniques using quantitative metrics (PSNR, SSIM) and qualitative visual comparisons.

Project Structure

The project is organized as follows:

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
│   └──  jpeg_compression.py
│   
│
├── restorations/             # Modules for image restoration algorithms
│   ├── __init__.py
│   ├── deep_learning/        # Deep learning based methods
│   │   ├── __init__.py
│   │   ├── deep_learning_denoiser.py # Applies the DRUNet model
│   │   └── drunet_model.py         # DRUNet architecture definition
│   └── traditional/          # Traditional algorithm-based methods
│       ├── __init__.py
│       └── Gaussian_Noise/   # Methods specific to Gaussian noise
│           ├── __init__.py
│           ├── median_filter.py
│           └── non_local_means.py
│
├── results/                  # Output directory for analysis results
│   └── for_GaussianNoise/    # Results specific to Gaussian noise analysis
│       ├── closeup_... .png         # Saved close-up comparison plots
│       ├── full_comparison_... .png # Saved full-size comparison plot
│       ├── metrics_chart_... .png   # Saved quantitative results chart
│       └── metrics_... .csv         # Saved quantitative results table
│
├── models/                   # Downloaded pre-trained model weights (created automatically)
│   └── drunet_color.pth
│
├── .gitignore                # Git ignore file (optional)
├── analysis.ipynb            # Jupyter Notebook for interactive exploration/visualization
├── calculate_metrics.py      # Main script to run the full analysis pipeline
└── requirements.txt          # Python dependencies


Setup

Clone the repository (if applicable):

git clone <your-repository-url>
cd Image-Distortion-Analysis


Create Python Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install Dependencies:

pip install -r requirements.txt


Add Original Images: Place your original .png or .jpg images into the data/original/ directory.

Usage

The primary way to run the analysis is using the calculate_metrics.py script.

Configure the Analysis: Open calculate_metrics.py and adjust the following parameters near the top:

NOISE_SIGMA: Set the desired standard deviation for the Gaussian noise (e.g., 25, 50).

IMAGES_FOR_VISUALIZATION: List the filenames of the images you want included in the visual comparison plots.

CROP_REGIONS: Important: Update the (x, y, width, height) coordinates in this dictionary for each image in IMAGES_FOR_VISUALIZATION to define the desired close-up area. Use an image editor to find appropriate coordinates.

RESULTS_DIR: Modify if you want results saved elsewhere (defaults to results/for_GaussianNoise).

Run the Script: Execute the script from your terminal in the project's root directory:

python calculate_metrics.py


What the script does:

Ensures the results and distorted directories exist.

Generates distorted images (with the specified NOISE_SIGMA) in data/distorted/gaussian_noise/ if they don't already exist.

Applies Median Filter, Non-local Means, and the DRUNet Deep Learning denoiser to the distorted images.

Calculates PSNR and SSIM for all methods compared to the original images.

Prints a table of the calculated metrics to the console.

Saves the full metrics table to a .csv file in the RESULTS_DIR.

Generates and saves a bar/line chart comparing the average PSNR and SSIM scores to the RESULTS_DIR.

Generates and saves the main 4-column visual comparison plot (using IMAGES_FOR_VISUALIZATION) to the RESULTS_DIR.

Generates and saves separate close-up comparison plots for each image specified (using CROP_REGIONS) to the RESULTS_DIR.

Note: The first time you run the script, it will download the pre-trained DRUNet model weights (drunet_color.pth) into the models/ directory. This requires an internet connection. Subsequent runs will use the cached model file.

Implemented Distortions

Gaussian Noise (gaussian_noise.py)

(Others like Gaussian Blur, JPEG Compression, etc., exist but the main script currently focuses on Gaussian Noise)

Implemented Restoration Methods (for Gaussian Noise)

Traditional:

Median Filter (median_filter.py)

Non-local Means (non_local_means.py)

Deep Learning:

DRUNet (Color Denoising Model) (deep_learning_denoiser.py)

Requirements

Python 3.x

OpenCV (opencv-python)

NumPy

scikit-image

Matplotlib

Pandas

PyTorch (torch, torchvision)

Install all via pip install -r requirements.txt.

(Optional: Add sections on Contributing, License, etc. if needed)