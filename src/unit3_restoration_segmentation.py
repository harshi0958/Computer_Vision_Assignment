import csv
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common import (
    ROOT, OUTPUT_DIR, load_image, save_image, save_comparison
)

OUT = OUTPUT_DIR / "unit3"
OUT.mkdir(parents=True, exist_ok=True)


def add_gaussian_noise(image, sigma=25):
    rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.04):
    rng = np.random.default_rng(42)
    noisy = image.copy()
    total = image.shape[0] * image.shape[1]
    count = int(total * amount / 2)

    ys = rng.integers(0, image.shape[0], count)
    xs = rng.integers(0, image.shape[1], count)
    noisy[ys, xs] = 255

    ys = rng.integers(0, image.shape[0], count)
    xs = rng.integers(0, image.shape[1], count)
    noisy[ys, xs] = 0
    return noisy


def calculate_metrics(reference, result):
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    psnr = peak_signal_noise_ratio(ref_gray, result_gray, data_range=255)
    ssim = structural_similarity(ref_gray, result_gray, data_range=255)
    return float(psnr), float(ssim)


def save_metrics(rows):
    path = OUT / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Noise Type", "Restoration Method", "PSNR (dB)", "SSIM"])
        writer.writerows(rows)
    print(f"Saved: {path.relative_to(ROOT)}")


def main():
    original = load_image()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 1. Noise generation
    gaussian_noisy = add_gaussian_noise(original)
    sp_noisy = add_salt_pepper_noise(original)

    save_image(gaussian_noisy, OUT / "01_gaussian_noise.png")
    save_image(sp_noisy, OUT / "02_salt_pepper_noise.png")
    save_comparison(
        [original, gaussian_noisy, sp_noisy],
        ["Original", "Gaussian Noise", "Salt-and-Pepper Noise"],
        OUT / "03_noise_comparison.png"
    )

    # 2. Image restoration
    gaussian_blur = cv2.GaussianBlur(gaussian_noisy, (5, 5), 1.2)
    gaussian_median = cv2.medianBlur(gaussian_noisy, 5)
    sp_median = cv2.medianBlur(sp_noisy, 5)
    sp_gaussian = cv2.GaussianBlur(sp_noisy, (5, 5), 1.2)

    restorations = [
        ("Gaussian Noise", "Gaussian Filter", gaussian_blur),
        ("Gaussian Noise", "Median Filter", gaussian_median),
        ("Salt-and-Pepper Noise", "Median Filter", sp_median),
        ("Salt-and-Pepper Noise", "Gaussian Filter", sp_gaussian),
    ]

    metric_rows = []
    for noise_name, method, restored in restorations:
        safe_name = method.lower().replace(" ", "_")
        noise_file = noise_name.lower().replace("-", "_").replace(" ", "_")
        save_image(restored, OUT / f"restored_{noise_file}_{safe_name}.png")
        psnr, ssim = calculate_metrics(original, restored)
        metric_rows.append([noise_name, method, round(psnr, 4), round(ssim, 6)])
        print(f"{noise_name} | {method}: PSNR={psnr:.3f} dB, SSIM={ssim:.4f}")

    save_comparison(
        [original, gaussian_noisy, gaussian_blur, gaussian_median,
         sp_noisy, sp_median],
        ["Original", "Gaussian Noise", "Gaussian Noise + Gaussian Filter",
         "Gaussian Noise + Median Filter", "Salt-and-Pepper Noise",
         "Salt-and-Pepper + Median Filter"],
        OUT / "04_restoration_comparison.png",
        cols=3
    )
    save_metrics(metric_rows)

    # 3. Binary thresholding
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    save_image(binary, OUT / "05_otsu_binary_threshold.png")

    # 4. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    save_comparison(
        [binary, erosion, dilation, opening, closing],
        ["Otsu Binary", "Erosion", "Dilation", "Opening", "Closing"],
        OUT / "06_morphological_operations.png",
        cols=3
    )
    save_image(erosion, OUT / "erosion.png")
    save_image(dilation, OUT / "dilation.png")
    save_image(opening, OUT / "opening.png")
    save_image(closing, OUT / "closing.png")

    # 5. Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    save_image(edges, OUT / "07_canny_edges.png")
    save_comparison(
        [gray, edges],
        ["Grayscale Image", "Canny Edges (100, 200)"],
        OUT / "08_canny_comparison.png",
        cols=2
    )

    # 6. Watershed segmentation
    # Otsu thresholding + morphological cleanup creates foreground seeds.
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    clean = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2
    )
    sure_bg = cv2.dilate(clean, kernel, iterations=3)
    distance = cv2.distanceTransform(clean, cv2.DIST_L2, 5)

    if distance.max() > 0:
        _, sure_fg = cv2.threshold(
            distance, 0.35 * distance.max(), 255, cv2.THRESH_BINARY
        )
        sure_fg = sure_fg.astype(np.uint8)
    else:
        sure_fg = np.zeros_like(gray)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_result = original.copy()
    markers = cv2.watershed(watershed_result, markers)
    watershed_result[markers == -1] = [0, 0, 255]

    save_image(watershed_result, OUT / "09_watershed_boundaries.png")
    save_image(
        cv2.applyColorMap(
            cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_TURBO
        ),
        OUT / "10_watershed_distance_map.png"
    )
    save_comparison(
        [original, clean, sure_fg, watershed_result],
        ["Original", "Cleaned Binary Mask", "Foreground Seeds", "Watershed Boundaries"],
        OUT / "11_watershed_comparison.png",
        cols=2
    )

    print("\nUnit 3 processing completed.")
    print(f"All results are saved in: {OUT}")


if __name__ == "__main__":
    main()
