# Computer Vision Assignment ΓÇö Units 3, 4 & 5

A practical Computer Vision project implemented using Python and OpenCV, covering image restoration, image segmentation, feature extraction, feature matching, object detection, and video motion analysis.

## Project Overview

This repository contains the Jupyter Notebook, Python source code, experimental outputs, evaluation metrics, and formatted assignment report for Computer Vision Units 3, 4, and 5.

## Topics Covered

### Unit 3 ΓÇö Image Restoration and Segmentation
- Gaussian and salt-and-pepper noise generation
- Gaussian and median filtering
- Image restoration comparisons and evaluation metrics
- Otsu thresholding
- Morphological operations: erosion, dilation, opening, and closing
- Canny edge detection
- Watershed segmentation

### Unit 4 ΓÇö Feature Extraction and Matching
- Histogram of Oriented Gradients (HOG)
- Scale-Invariant Feature Transform (SIFT)
- Keypoint detection and visualization
- Feature matching between images
- Comparative matching results

### Unit 5 ΓÇö Object Detection and Motion Analysis
- Haar Cascade face/object detection
- YOLOv8 object detection
- Frame differencing
- Background subtraction using MOG2
- Optical flow analysis
- Comparative motion-analysis metrics

## Repository Structure

```text
CV_Units_3_4_5_Assignment/
Γö£ΓöÇΓöÇ data/
Γöé   ΓööΓöÇΓöÇ images/                 # Input images
Γö£ΓöÇΓöÇ outputs/
Γöé   Γö£ΓöÇΓöÇ unit3/                  # Restoration and segmentation results
Γöé   Γö£ΓöÇΓöÇ unit4/                  # Feature extraction and matching results
Γöé   ΓööΓöÇΓöÇ unit5/                  # Detection and motion-analysis results
Γö£ΓöÇΓöÇ src/
Γöé   Γö£ΓöÇΓöÇ common.py
Γöé   ΓööΓöÇΓöÇ unit3_restoration_segmentation.py
Γö£ΓöÇΓöÇ CV_Units_3_4_5_Assignment.ipynb
Γö£ΓöÇΓöÇ COMPUTER_VISION_ASSIGNMENT_REPORT.docx
Γö£ΓöÇΓöÇ outputs.zip
Γö£ΓöÇΓöÇ requirements.txt
Γö£ΓöÇΓöÇ .gitignore
ΓööΓöÇΓöÇ README.md
```

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Jupyter Notebook
- YOLOv8

Refer to `requirements.txt` for the project's Python dependencies.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/harshi0958/Computer_Vision_Assignment.git
cd Computer_Vision_Assignment
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook CV_Units_3_4_5_Assignment.ipynb
```

Run the notebook cells in sequence. Ensure the required input images and any model weights needed for the relevant cells are available.

## Results and Evaluation

Generated visualizations and evaluation files are organized by unit in the `outputs/` directory. These include image restoration comparisons, feature-matching results, object-detection visualizations, motion-analysis frames, and CSV-based metrics.

The `outputs.zip` archive provides a bundled copy of the generated outputs.

## Assignment Report

The formatted report is available at:

`COMPUTER_VISION_ASSIGNMENT_REPORT.docx`

It documents the assignment work and should be consulted alongside the notebook and generated results.

## Notes

- Use a compatible Python environment and install the dependencies listed in `requirements.txt`.
- YOLOv8 inference may require model weights and additional downloads, depending on the execution environment.
- Input videos are not included in the documented repository structure; provide the required video inputs if running video-based experiments.
- Results can vary with input images, model weights, library versions, and execution settings.

## Repository

GitHub: https://github.com/harshi0958/Computer_Vision_Assignment
