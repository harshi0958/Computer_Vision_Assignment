from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT_DIR = ROOT / "outputs"

def load_image(filename="original_photo.jpg"):
    path = IMAGE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            "Copy your original photo into data/images/"
            " and name it original_photo.jpg."
        )
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {path}")
    return image

def save_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not save image: {output_path}")
    print(f"Saved: {output_path.relative_to(ROOT)}")

def save_comparison(images, titles, output_path, cols=3):
    if len(images) != len(titles):
        raise ValueError("Each image must have a corresponding title.")

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, image, title in zip(axes, images, titles):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison: {output_path.relative_to(ROOT)}")
