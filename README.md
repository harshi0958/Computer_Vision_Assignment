# Computer Vision Assignment — Units 3, 4 & 5

### Image Processing • Feature Extraction • Object Detection • Motion Analysis

A practical implementation of computer vision and digital image processing techniques using Python, OpenCV, and machine learning libraries.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-111F68)](https://docs.ultralytics.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## 📌 Project Overview

This repository contains practical experiments covering key concepts from **Computer Vision and Digital Image Processing**, organised into Units 3, 4, and 5.

The project explores how images and videos can be processed, enhanced, segmented, analysed, and interpreted using traditional computer vision algorithms and deep-learning-based object detection.

The implementations include image restoration, segmentation, feature extraction, feature matching, object detection, and video motion analysis. Experimental outputs and image-quality metrics are used to examine the behaviour and limitations of different techniques.

## 🎯 Objectives

- Implement image restoration and noise-reduction techniques.
- Evaluate image quality using PSNR and SSIM.
- Perform image thresholding, morphological processing, and segmentation.
- Extract and match visual features using HOG and SIFT.
- Compare Haar Cascade and YOLO-based object detection.
- Analyse video motion using frame differencing, MOG2, and Lucas–Kanade optical flow.
- Generate output images and comparison results for analysis.

## 📚 Topics Covered

### Unit 3 — Image Restoration and Segmentation

| Technique | Purpose |
|---|---|
| Gaussian Noise | Study image degradation caused by Gaussian noise |
| Salt-and-Pepper Noise | Simulate impulse noise in images |
| Gaussian Filtering | Reduce noise through Gaussian smoothing |
| Median Filtering | Reduce impulse noise while preserving edges |
| PSNR and SSIM | Evaluate image restoration quality |
| Otsu Thresholding | Separate foreground and background automatically |
| Morphological Operations | Refine binary regions and remove small irregularities |
| Canny Edge Detection | Identify significant image boundaries |
| Watershed Segmentation | Separate connected or touching image regions |

### Unit 4 — Feature Extraction and Feature Matching

| Technique | Purpose |
|---|---|
| HOG (Histogram of Oriented Gradients) | Represent image shape and gradient patterns |
| SIFT (Scale-Invariant Feature Transform) | Detect and describe distinctive image features |
| SIFT Feature Matching | Identify corresponding features across images |
| Rotation Experiments | Study feature matching under image rotation |
| Resizing Experiments | Examine matching under scale changes |

### Unit 5 — Object Detection and Video Motion Analysis

| Technique | Purpose |
|---|---|
| Haar Cascade | Detect objects supported by a trained cascade classifier |
| YOLO | Detect supported object categories using deep learning |
| Frame Differencing | Identify changes between consecutive video frames |
| MOG2 Background Subtraction | Estimate foreground regions against an adaptive background |
| Lucas–Kanade Optical Flow | Estimate local motion by tracking image features |

## 🛠️ Tech Stack

- **Python** — Primary programming language
- **OpenCV** — Image processing, feature detection, and video analysis
- **NumPy** — Numerical computation and image-array operations
- **Matplotlib** — Visualisation of images and experimental results
- **Scikit-image** — Image-processing utilities and quality metrics
- **Ultralytics YOLO** — Deep-learning-based object detection
- **Jupyter Notebook** — Interactive implementation and experimentation

## 📊 Results and Analysis

The experiments produce visual outputs and measurements that help compare the implemented techniques.

Examples of the evaluated results include:

- Noisy images and their filtered versions.
- PSNR and SSIM measurements for image restoration.
- Thresholded images, morphological outputs, edge maps, and watershed segmentation.
- HOG visualisations, SIFT keypoints, and feature-matching results.
- Haar Cascade and YOLO object-detection outputs.
- Frame-difference masks, MOG2 foreground masks, and optical-flow visualisations.

The report also discusses the advantages, limitations, and practical applications of the techniques. Results should be interpreted according to the input data, algorithm settings, and available evaluation metrics.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/harshi0958/Computer_Vision_Assignment.git
cd Computer_Vision_Assignment
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

Install the libraries used by the project:

```bash
pip install opencv-python numpy matplotlib scikit-image ultralytics jupyter
```

If the repository contains a `requirements.txt` file, you can install its pinned dependencies instead:

```bash
pip install -r requirements.txt
```

### 4. Run the experiments

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the relevant notebook in the repository and execute its cells in sequence. Ensure that the required input images, video, model weights, and supporting files are available at the paths expected by the implementation.

**Note:** YOLO model downloads or additional model files may be required depending on the configuration used.

## 📁 Repository Contents

The repository contains the assignment's implementation files, supporting input data, and selected experimental outputs.

The exact directory and notebook names should be taken from the current repository. Refer to the files committed to GitHub for the authoritative project structure.

## 🎓 Academic Information

| Detail | Information |
|---|---|
| Student | Harshit Jariwala |
| Programme | Master of Computer Applications (MCA) |
| Subject | Computer Vision and Digital Image Processing |
| Assignment | Units 3, 4 and 5 |
| Submitted to | Rahul Inchal |
| Submission Date | 20 October 2026 |

## 🔮 Applications

The techniques explored in this assignment are relevant to:

- Image enhancement and restoration
- Medical and scientific image analysis
- Object and feature recognition
- Video surveillance and motion detection
- Traffic monitoring and video analytics
- Robotics and visual navigation
- Industrial inspection and automation

## 👨‍💻 Author

**Harshit Jariwala**

- GitHub: [@harshi0958](https://github.com/harshi0958)
- Project Repository: [Computer_Vision_Assignment](https://github.com/harshi0958/Computer_Vision_Assignment)

---

*Developed as an academic practical assignment to explore computer vision algorithms, image-processing techniques, and video-analysis methods using Python.*
