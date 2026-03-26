import cv2
import os
import argparse
from video_loader import VideoLoader
from tracker import Tracker
from trajectory import TrajectoryTracker


def main():
    parser = argparse.ArgumentParser(description="Moving Object Detection and Tracking")
    parser.add_argument("--video", type=str, default="dataset/VIRAT/video1.mp4",
                        help="Path to input video relative to project root")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output video relative to project root")
    parser.add_argument("--detector", type=str, default="yolo", choices=["yolo", "mog2"],
                        help="Detection backend: 'yolo' (YOLOv8, recommended) or 'mog2' (legacy)")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model weight file (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="YOLO NMS IOU threshold")
    args = parser.parse_args()

    # ---- paths -----------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(base_dir, args.video.replace("/", os.sep))
    output_dir = os.path.join(base_dir, args.output_dir.replace("/", os.sep))
    os.makedirs(output_dir, exist_ok=True)

    # ---- video loader ----------------------------------------------------
    print(f"Loading video from: {video_path}")
    try:
        loader = VideoLoader(video_path)
    except Exception as e:
        print(f"Error loading video: {e}")
        print("Please ensure you have placed a valid video file at the specified path.")
        return

    width, height, fps = loader.get_properties()

    # ---- video writer ----------------------------------------------------
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ---- detector --------------------------------------------------------
    if args.detector == "yolo":
        from yolo_detector import YOLODetector
        detector = YOLODetector(model_name=args.yolo_model, conf=args.conf, iou=args.iou)
        use_yolo = True
        print(f"[main] Using YOLOv8 detector ({args.yolo_model})")
    else:
        from motion_detection import MotionDetector
        detector = MotionDetector()
        use_yolo = False
        print("[main] Using MOG2 motion detector (legacy)")

    # ---- tracker + trajectory --------------------------------------------
    tracker = Tracker()
    trajectory_tracker = TrajectoryTracker(max_points=50)

    print("Starting processing. Press 'q' to quit.")
    frame_count = 0

    while True:
        ret, frame = loader.read_frame()
        if not ret:
            print("Reached end of video.")
            break

        # ---- detection ---------------------------------------------------
        if use_yolo:
            detections = detector.detect(frame)
        else:
            detections, fg_mask = detector.apply(frame)

        # ---- tracking ----------------------------------------------------
        tracks = tracker.update(detections, frame)

        # ---- drawing -----------------------------------------------------
        processed_frame = frame.copy()
        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            active_ids.add(track_id)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Update trajectory
            trajectory_tracker.update(track_id, (center_x, center_y))

            # Draw trajectory
            trajectory_tracker.draw(processed_frame, track_id, color=(0, 255, 255), thickness=2)

            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label — include class name from YOLO when available
            det_class = track.det_class if hasattr(track, 'det_class') and track.det_class else "Obj"
            label = f"{det_class} {track_id}"
            cv2.putText(processed_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Cleanup old trajectories
        trajectory_tracker.cleanup(active_ids)

        # Save to output
        out.write(processed_frame)

        # Overlay HUD
        cv2.putText(processed_frame, f"Frame: {frame_count}  Tracks: {len(active_ids)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display
        cv2.imshow('Object Detection + Tracking', processed_frame)
        if not use_yolo:
            cv2.imshow('Foreground Mask', fg_mask)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User interrupted processing.")
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # ---- cleanup ---------------------------------------------------------
    loader.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
