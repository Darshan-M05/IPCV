"""
Step 3: Feature Extraction
==========================
Reads preprocessed images, extracts feature vectors (HOG or YOLO embeddings),
saves as .npy, and generates a t-SNE visualisation.

Usage:
    python src/feature_extraction.py
    python src/feature_extraction.py --method hog
    python src/feature_extraction.py --method yolo
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

BASE_DIR = os.path.dirname(_SRC_DIR)


# ── HOG feature extractor ────────────────────────────────────────────────────

def extract_hog_features(image, target_size=(128, 128)):
    """Extract HOG descriptor from a single BGR image.

    Uses OpenCV's HOGDescriptor with parameters suited for
    person/vehicle classification.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)

    hog = cv2.HOGDescriptor(
        _winSize=target_size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    features = hog.compute(resized)
    return features.flatten()


# ── YOLO embedding extractor ─────────────────────────────────────────────────

def extract_yolo_embeddings(image, model):
    """Extract feature embeddings from YOLOv8 backbone using the embed API."""
    results = model.predict(image, verbose=False, embed=[12])  # layer 12 features
    # If embed returns feature maps, flatten and use.  Fallback to zeros.
    if results and hasattr(results[0], 'embed') and results[0].embed is not None:
        embedding = results[0].embed[0].cpu().numpy().flatten()
    else:
        # Fallback: use detection confidence vector as a simple embedding
        embedding = np.zeros(256)
    return embedding


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_feature_extraction(
    input_dir,
    output_dir,
    method="hog",
    model_name="yolov8s.pt",
    max_images=None,
):
    """Extract features and save as .npy files."""

    os.makedirs(output_dir, exist_ok=True)

    categories = ["person", "vehicle", "background"]
    label_map = {"person": 0, "vehicle": 1, "background": 2}

    all_features = []
    all_labels = []

    # Load YOLO model if needed
    yolo_model = None
    if method == "yolo":
        from ultralytics import YOLO
        import torch
        yolo_model = YOLO(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        yolo_model.to(device)
        print(f"[Features] YOLO model on {device}")

    for cat in categories:
        folder = os.path.join(input_dir, cat)
        images = sorted(glob.glob(os.path.join(folder, "*.jpg")))

        if max_images:
            images = images[:max_images]

        if not images:
            print(f"  [{cat}] No images found, skipping.")
            continue

        print(f"  [{cat}] Extracting {method.upper()} features from {len(images)} images...")

        for img_path in tqdm(images, desc=f"  {cat}", unit="img", leave=False):
            img = cv2.imread(img_path)
            if img is None:
                continue

            if method == "hog":
                feat = extract_hog_features(img)
            else:
                feat = extract_yolo_embeddings(img, yolo_model)

            all_features.append(feat)
            all_labels.append(label_map[cat])

    # Save
    features_arr = np.array(all_features)
    labels_arr = np.array(all_labels)

    feat_path = os.path.join(output_dir, "feature_vectors.npy")
    lbl_path = os.path.join(output_dir, "labels.npy")
    np.save(feat_path, features_arr)
    np.save(lbl_path, labels_arr)

    print(f"\n[Features] Saved {features_arr.shape[0]} vectors, dim={features_arr.shape[1]}")
    print(f"  feature_vectors.npy: {feat_path}")
    print(f"  labels.npy:          {lbl_path}")

    # ── t-SNE visualisation ──────────────────────────────────────────────
    generate_tsne_plot(features_arr, labels_arr, categories, output_dir)

    return features_arr, labels_arr


def generate_tsne_plot(features, labels, category_names, output_dir):
    """Create a t-SNE scatter plot of the feature space."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print("  Generating t-SNE visualisation...")

    # Subsample if too many points
    max_points = 2000
    if len(features) > max_points:
        idx = np.random.choice(len(features), max_points, replace=False)
        features = features[idx]
        labels = labels[idx]

    # Reduce to 2D
    perplexity = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colours = ["#f38ba8", "#a6e3a1", "#89b4fa"]  # catppuccin reds, greens, blues

    for i, cat in enumerate(category_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=colours[i], label=cat, alpha=0.6, s=20)

    ax.set_title("Feature Space Visualisation (t-SNE)", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "feature_tsne.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  t-SNE plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline")
    parser.add_argument("--input", type=str,
                        default=os.path.join(BASE_DIR, "processed_dataset"),
                        help="Input folder (from preprocessing.py)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(BASE_DIR, "output", "features"),
                        help="Output folder for features and plots")
    parser.add_argument("--method", type=str, default="hog",
                        choices=["hog", "yolo"],
                        help="Feature extraction method")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="YOLOv8 model (for --method yolo)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images per category (for testing)")
    args = parser.parse_args()

    run_feature_extraction(
        input_dir=args.input,
        output_dir=args.output,
        method=args.method,
        model_name=args.model,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
