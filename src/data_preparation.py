"""
Step 1: Dataset Frame Extraction & Auto-Labelling
==================================================
Reads all VIRAT videos, extracts frames at a configurable interval,
runs YOLOv8 to classify each frame, and saves them into labelled folders.

Usage:
    python src/data_preparation.py
    python src/data_preparation.py --interval 10 --max-videos 5
"""

import os
import sys
import glob
import argparse
import cv2
from tqdm import tqdm

# Ensure src/ is importable
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

BASE_DIR = os.path.dirname(_SRC_DIR)

# COCO class-ID → category mapping used for folder routing
_PERSON_CLASSES = {0}                     # person
_VEHICLE_CLASSES = {1, 2, 3, 5, 7}       # bicycle, car, motorcycle, bus, truck


def classify_frame(results):
    """Decide the label for a frame based on YOLO detections.

    Priority: person > vehicle > background
    """
    has_person = False
    has_vehicle = False

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id in _PERSON_CLASSES:
                has_person = True
            elif cls_id in _VEHICLE_CLASSES:
                has_vehicle = True

    if has_person:
        return "person"
    elif has_vehicle:
        return "vehicle"
    else:
        return "background"


def extract_frames(
    dataset_dir,
    output_dir,
    model_name="yolov8s.pt",
    interval=5,
    img_size=640,
    max_videos=None,
):
    """Main extraction pipeline."""

    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DataPrep] Device: {device}")
    if device == "cpu":
        print("[DataPrep] WARNING: Running on CPU — this will be slow.")
        print("[DataPrep] Install CUDA PyTorch: pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124")

    model = YOLO(model_name)
    model.to(device)

    # Prepare output folders
    categories = ["person", "vehicle", "background"]
    for cat in categories:
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    # Gather videos
    video_dir = os.path.join(dataset_dir, "VIRAT")
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if max_videos:
        videos = videos[:max_videos]

    print(f"[DataPrep] Found {len(videos)} videos, interval={interval}")

    counters = {cat: 0 for cat in categories}
    total_saved = 0

    for vid_idx, video_path in enumerate(videos):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_extract = total_frames // interval

        print(f"\n[{vid_idx + 1}/{len(videos)}] {video_name} ({total_frames} frames, "
              f"extracting ~{frames_to_extract})")

        pbar = tqdm(total=frames_to_extract, desc=f"  {video_name}", unit="frame")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval != 0:
                frame_idx += 1
                continue

            # Resize to target
            resized = cv2.resize(frame, (img_size, img_size))

            # Run YOLO (use imgsz matching the already-resized frame)
            results = model.predict(resized, conf=0.3, imgsz=img_size, verbose=False)

            # Classify
            label = classify_frame(results)

            # Save
            fname = f"{video_name}_f{frame_idx:06d}.jpg"
            save_path = os.path.join(output_dir, label, fname)
            cv2.imwrite(save_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

            counters[label] += 1
            total_saved += 1
            pbar.update(1)
            frame_idx += 1

        pbar.close()
        cap.release()

    print(f"\n[DataPrep] Done! Total frames saved: {total_saved}")
    for cat, count in counters.items():
        print(f"  {cat:12s}: {count}")

    return counters


def main():
    parser = argparse.ArgumentParser(description="Dataset Frame Extraction & Auto-Labelling")
    parser.add_argument("--dataset", type=str, default=os.path.join(BASE_DIR, "dataset"),
                        help="Path to dataset folder")
    parser.add_argument("--output", type=str, default=os.path.join(BASE_DIR, "dataset_frames"),
                        help="Output folder for extracted frames")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="YOLOv8 model weight file")
    parser.add_argument("--interval", type=int, default=5,
                        help="Extract every Nth frame")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Resize frames to this size")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit number of videos (for testing)")
    args = parser.parse_args()

    extract_frames(
        dataset_dir=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        interval=args.interval,
        img_size=args.img_size,
        max_videos=args.max_videos,
    )


if __name__ == "__main__":
    main()
