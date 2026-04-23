"""
Step 2: Image Preprocessing Pipeline
=====================================
Reads extracted frames from dataset_frames/, applies resize + denoise + CLAHE,
saves to processed_dataset/, and generates before/after samples.

Usage:
    python src/preprocessing.py
    python src/preprocessing.py --size 224 --samples 10
"""

import os
import sys
import glob
import random
import argparse
import cv2
import numpy as np
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

BASE_DIR = os.path.dirname(_SRC_DIR)


def preprocess_image(image, target_size=224):
    """
    Apply the full preprocessing pipeline to a single image.

    [IPCV Syllabus Mapping - UNIT I: Image Processing]
      - Geometric Transformations: Resizing pixel matrices.
      - Linear Filtering: Gaussian kernels to convolute matrices.
      - Point Operators: CLAHE performing non-linear contrast adjustments.

    Returns:
        processed : preprocessed BGR image
    """
    # 1. Geometric Transformation: Resize (Affine bounds)
    resized = cv2.resize(image, (target_size, target_size))

    # 2. Linear Filtering: Gaussian Denoise (5x5 spatial domain)
    # Suppresses high-frequency artifacts safely.
    denoised = cv2.GaussianBlur(resized, (5, 5), 0)

    # 3. Point Operators: CLAHE on luminance (L channel)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
    processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return processed


def make_before_after(before, after, target_h=300):
    """Create a side-by-side comparison image with labels."""
    # Resize both to same height for clean display
    h1, w1 = before.shape[:2]
    h2, w2 = after.shape[:2]
    before_disp = cv2.resize(before, (int(w1 * target_h / h1), target_h))
    after_disp = cv2.resize(after, (int(w2 * target_h / h2), target_h))

    # Add labels
    cv2.putText(before_disp, "BEFORE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(after_disp, "AFTER", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Stack horizontally
    combined = np.hstack([before_disp, after_disp])
    return combined


def run_preprocessing(
    input_dir,
    output_dir,
    sample_dir,
    target_size=224,
    num_samples=10,
):
    """Main preprocessing pipeline."""

    categories = ["person", "vehicle", "background"]
    all_pairs = []  # (before_path, after_path) for sampling

    for cat in categories:
        src_folder = os.path.join(input_dir, cat)
        dst_folder = os.path.join(output_dir, cat)
        os.makedirs(dst_folder, exist_ok=True)

        images = sorted(glob.glob(os.path.join(src_folder, "*.jpg")))
        if not images:
            print(f"  [{cat}] No images found, skipping.")
            continue

        print(f"  [{cat}] Processing {len(images)} images...")

        for img_path in tqdm(images, desc=f"  {cat}", unit="img", leave=False):
            img = cv2.imread(img_path)
            if img is None:
                continue

            processed = preprocess_image(img, target_size=target_size)

            fname = os.path.basename(img_path)
            out_path = os.path.join(dst_folder, fname)
            cv2.imwrite(out_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

            all_pairs.append((img_path, out_path))

    # ── Generate before/after samples ────────────────────────────────────
    os.makedirs(sample_dir, exist_ok=True)
    num_samples = min(num_samples, len(all_pairs))

    if num_samples > 0:
        samples = random.sample(all_pairs, num_samples)
        print(f"\n  Generating {num_samples} before/after samples...")

        for i, (before_path, after_path) in enumerate(samples):
            before_img = cv2.imread(before_path)
            after_img = cv2.imread(after_path)
            if before_img is None or after_img is None:
                continue

            comparison = make_before_after(before_img, after_img)
            save_path = os.path.join(sample_dir, f"sample_{i+1:02d}.jpg")
            cv2.imwrite(save_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

    total = len(all_pairs)
    print(f"\n[Preprocessing] Done! {total} images processed.")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {sample_dir}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Image Preprocessing Pipeline")
    parser.add_argument("--input", type=str,
                        default=os.path.join(BASE_DIR, "dataset_frames"),
                        help="Input folder (from data_preparation.py)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(BASE_DIR, "processed_dataset"),
                        help="Output folder for preprocessed images")
    parser.add_argument("--samples-dir", type=str,
                        default=os.path.join(BASE_DIR, "sample_before_after"),
                        help="Output folder for before/after comparisons")
    parser.add_argument("--size", type=int, default=224,
                        help="Target image size (224 or 640)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of before/after samples to generate")
    args = parser.parse_args()

    run_preprocessing(
        input_dir=args.input,
        output_dir=args.output,
        sample_dir=args.samples_dir,
        target_size=args.size,
        num_samples=args.samples,
    )


if __name__ == "__main__":
    main()
