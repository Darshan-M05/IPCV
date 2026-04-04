"""
Step 4: Evaluation Metrics & Visualization
==========================================
Runs the detection + tracking pipeline on a video, computes metrics
(precision, recall, F1, ID switches, tracking accuracy, FPS), and
generates charts and reports.

Usage:
    python src/evaluation.py
    python src/evaluation.py --video dataset/VIRAT/video10.mp4 --max-frames 500
"""

import os
import sys
import json
import csv
import time
import argparse
import cv2
import numpy as np

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

BASE_DIR = os.path.dirname(_SRC_DIR)


def run_evaluation(
    video_path,
    output_dir,
    model_name="yolov8s.pt",
    conf=0.25,
    iou=0.4,
    max_frames=None,
):
    """Run detection + tracking and collect per-frame metrics."""

    from yolo_detector import YOLODetector
    from tracker import Tracker
    from video_loader import VideoLoader

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Eval] Video: {video_path}")
    print(f"[Eval] Model: {model_name}")

    loader = VideoLoader(video_path)
    width, height, fps = loader.get_properties()
    total_vid_frames = loader.total_frames

    detector = YOLODetector(model_name=model_name, conf=conf, iou=iou, imgsz=640)
    tracker = Tracker()

    # ── Per-frame metric collection ──────────────────────────────────────
    frame_times = []
    detection_counts = []
    track_counts = []
    all_track_ids_per_frame = []
    seen_ids = set()
    id_history = {}            # track_id → list of frame indices
    frame_idx = 0

    print("[Eval] Processing frames...")

    while True:
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = loader.read_frame()
        if not ret:
            break

        t0 = time.time()
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        elapsed = time.time() - t0

        frame_times.append(elapsed)
        detection_counts.append(len(detections))

        confirmed_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            confirmed_ids.add(tid)

            if tid not in id_history:
                id_history[tid] = []
            id_history[tid].append(frame_idx)

        track_counts.append(len(confirmed_ids))
        all_track_ids_per_frame.append(confirmed_ids)
        seen_ids.update(confirmed_ids)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}...")

    loader.release()
    num_frames = frame_idx
    print(f"[Eval] Processed {num_frames} frames.")

    # ── Compute metrics ──────────────────────────────────────────────────

    # FPS statistics
    fps_values = [1.0 / t if t > 0 else 0 for t in frame_times]
    avg_fps = np.mean(fps_values) if fps_values else 0
    min_fps = np.min(fps_values) if fps_values else 0
    max_fps = np.max(fps_values) if fps_values else 0

    # Detection statistics
    avg_detections = np.mean(detection_counts) if detection_counts else 0
    total_detections = int(np.sum(detection_counts))

    # Tracking statistics
    avg_tracks = np.mean(track_counts) if track_counts else 0
    total_unique_ids = len(seen_ids)

    # ID switches: count gaps in each track's frame history
    # A gap means the track disappeared and a new ID was likely assigned
    id_switches_estimate = 0
    for tid, frames in id_history.items():
        for i in range(1, len(frames)):
            if frames[i] - frames[i - 1] > 1:
                id_switches_estimate += 1

    max_concurrent = max(track_counts) if track_counts else 0

    # Tracking accuracy: fraction of frames with at least one confirmed track
    frames_with_tracks = sum(1 for c in track_counts if c > 0)
    tracking_accuracy = frames_with_tracks / num_frames if num_frames > 0 else 0
    tracking_accuracy = float(np.clip(tracking_accuracy, 0.0, 1.0))

    # ── Precision / Recall / F1 (per-frame TP/FP/FN estimation) ──────────
    #
    # Without ground-truth annotations we estimate:
    #   TP = detections that became confirmed tracks  (matched)
    #   FP = detections that did NOT become tracks    (false alarms)
    #   FN = confirmed tracks without a detection     (missed by detector)
    #
    # Per frame:
    #   TP_i = min(detections_i, confirmed_tracks_i)
    #   FP_i = max(detections_i - confirmed_tracks_i, 0)
    #   FN_i = max(confirmed_tracks_i - detections_i, 0)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    for det_c, trk_c in zip(detection_counts, track_counts):
        tp = min(det_c, trk_c)
        fp = max(det_c - trk_c, 0)
        fn = max(trk_c - det_c, 0)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Validate: clip to [0, 1]
    precision = float(np.clip(precision, 0.0, 1.0))
    recall    = float(np.clip(recall, 0.0, 1.0))
    f1        = float(np.clip(f1, 0.0, 1.0))

    assert 0.0 <= precision <= 1.0, f"Precision out of range: {precision}"
    assert 0.0 <= recall    <= 1.0, f"Recall out of range: {recall}"
    assert 0.0 <= f1        <= 1.0, f"F1 out of range: {f1}"

    # ── Build report ─────────────────────────────────────────────────────

    report = {
        "video": os.path.basename(video_path),
        "model": model_name,
        "total_frames": num_frames,
        "detection": {
            "total_detections": total_detections,
            "avg_detections_per_frame": round(float(avg_detections), 2),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
        },
        "tracking": {
            "total_unique_ids": total_unique_ids,
            "max_concurrent_tracks": max_concurrent,
            "avg_tracks_per_frame": round(float(avg_tracks), 2),
            "id_switches_estimate": id_switches_estimate,
            "tracking_accuracy": round(float(tracking_accuracy), 4),
        },
        "performance": {
            "avg_fps": round(float(avg_fps), 2),
            "min_fps": round(float(min_fps), 2),
            "max_fps": round(float(max_fps), 2),
        },
    }

    # Save JSON
    json_path = os.path.join(output_dir, "performance_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {json_path}")

    # Save CSV
    csv_path = os.path.join(output_dir, "performance_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Video", report["video"]])
        writer.writerow(["Model", report["model"]])
        writer.writerow(["Total Frames", report["total_frames"]])
        writer.writerow(["Total Detections", report["detection"]["total_detections"]])
        writer.writerow(["Avg Detections/Frame", report["detection"]["avg_detections_per_frame"]])
        writer.writerow(["Precision", report["detection"]["precision"]])
        writer.writerow(["Recall", report["detection"]["recall"]])
        writer.writerow(["F1 Score", report["detection"]["f1_score"]])
        writer.writerow(["Unique Track IDs", report["tracking"]["total_unique_ids"]])
        writer.writerow(["Max Concurrent Tracks", report["tracking"]["max_concurrent_tracks"]])
        writer.writerow(["ID Switches (est.)", report["tracking"]["id_switches_estimate"]])
        writer.writerow(["Tracking Accuracy", report["tracking"]["tracking_accuracy"]])
        writer.writerow(["Avg FPS", report["performance"]["avg_fps"]])
        writer.writerow(["Min FPS", report["performance"]["min_fps"]])
        writer.writerow(["Max FPS", report["performance"]["max_fps"]])
    print(f"  Table:  {csv_path}")

    # ── Generate charts ──────────────────────────────────────────────────
    generate_charts(
        fps_values, detection_counts, track_counts,
        report, output_dir, num_frames,
    )

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Precision:         {report['detection']['precision']:.4f}")
    print(f"  Recall:            {report['detection']['recall']:.4f}")
    print(f"  F1 Score:          {report['detection']['f1_score']:.4f}")
    print(f"  Tracking Accuracy: {report['tracking']['tracking_accuracy']:.4f}")
    print(f"  ID Switches (est): {report['tracking']['id_switches_estimate']}")
    print(f"  Avg FPS:           {report['performance']['avg_fps']:.1f}")
    print("=" * 50)

    return report


def generate_charts(fps_values, det_counts, track_counts, report, output_dir, num_frames):
    """Generate 3 analysis charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames_x = list(range(len(fps_values)))

    # ── 1. Detection accuracy bar chart ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Precision", "Recall", "F1 Score"]
    values = [
        report["detection"]["precision"],
        report["detection"]["recall"],
        report["detection"]["f1_score"],
    ]
    colours = ["#89b4fa", "#a6e3a1", "#f9e2af"]
    bars = ax.bar(metrics, values, color=colours, edgecolor="#313244", linewidth=1.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Detection Accuracy Metrics", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "detection_accuracy.png"), dpi=150)
    plt.close(fig)

    # ── 2. FPS over time ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    # Smooth with rolling average
    window = min(20, len(fps_values))
    if window > 1:
        smoothed = np.convolve(fps_values, np.ones(window) / window, mode="valid")
        ax.plot(range(len(smoothed)), smoothed, color="#89b4fa", linewidth=1.5,
                label=f"Rolling avg (w={window})")
    ax.axhline(y=report["performance"]["avg_fps"], color="#a6e3a1",
               linestyle="--", label=f'Mean: {report["performance"]["avg_fps"]:.1f}')
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("FPS", fontsize=12)
    ax.set_title("Frame Processing Speed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fps_chart.png"), dpi=150)
    plt.close(fig)

    # ── 3. Tracking accuracy over time ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(frames_x, track_counts, alpha=0.3, color="#a6e3a1")
    ax.plot(frames_x, track_counts, color="#a6e3a1", linewidth=1, label="Active Tracks")
    ax2 = ax.twinx()
    ax2.plot(frames_x, det_counts, color="#f38ba8", linewidth=1, alpha=0.7,
             label="Raw Detections")
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Active Tracks", fontsize=12, color="#a6e3a1")
    ax2.set_ylabel("Raw Detections", fontsize=12, color="#f38ba8")
    ax.set_title("Tracking Performance Over Time", fontsize=14, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tracking_chart.png"), dpi=150)
    plt.close(fig)

    print(f"  Charts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluation Metrics & Visualization")
    parser.add_argument("--video", type=str,
                        default=os.path.join(BASE_DIR, "dataset", "VIRAT", "video10.mp4"),
                        help="Video file to evaluate on")
    parser.add_argument("--output", type=str,
                        default=os.path.join(BASE_DIR, "output", "analysis"),
                        help="Output folder for reports and charts")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="YOLOv8 model weight file")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit frames to process (for testing)")
    args = parser.parse_args()

    run_evaluation(
        video_path=args.video,
        output_dir=args.output,
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
