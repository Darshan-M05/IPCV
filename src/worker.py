import cv2
import os
import sys
import time
import traceback
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage


class VideoWorker(QThread):
    """
    Background worker that runs the YOLOv8 + DeepSORT + Trajectory pipeline.

    Signals
    -------
    frame_ready(QImage, dict)  – emitted every processed frame with stats
    finished(str)              – emitted when video ends or is stopped
    error(str)                 – emitted on unrecoverable errors
    status(str)                – status messages (e.g. "Loading model...")
    """

    frame_ready = pyqtSignal(QImage, dict)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, video_path, output_dir, model_name="yolov8s.pt",
                 conf=0.25, iou=0.4, imgsz=1280, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        self._stop_flag = False
        self._mutex = QMutex()

    # ── control ───────────────────────────────────────────────────────────

    def stop(self):
        """Thread-safe request to stop processing."""
        self._mutex.lock()
        self._stop_flag = True
        self._mutex.unlock()

    def _is_stopped(self):
        self._mutex.lock()
        val = self._stop_flag
        self._mutex.unlock()
        return val

    # ── main entry ────────────────────────────────────────────────────────

    def run(self):
        """Overrides QThread.run — executes in the background thread."""
        try:
            self._run_pipeline()
        except Exception:
            tb = traceback.format_exc()
            print(f"[VideoWorker] ERROR:\n{tb}", file=sys.stderr)
            self.error.emit(tb)

    # ── pipeline ──────────────────────────────────────────────────────────

    def _run_pipeline(self):
        # Ensure src/ is on sys.path so local imports work
        src_dir = os.path.dirname(os.path.abspath(__file__))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from video_loader import VideoLoader
        from yolo_detector import YOLODetector
        from tracker import Tracker
        from trajectory import TrajectoryTracker

        # ── video I/O ─────────────────────────────────────────────────────
        self.status.emit("Loading video...")
        loader = VideoLoader(self.video_path)
        width, height, fps = loader.get_properties()

        os.makedirs(self.output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_path = os.path.join(self.output_dir, f"{video_name}_tracked.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # ── model init ────────────────────────────────────────────────────
        self.status.emit("Loading YOLOv8 model (first run downloads weights)...")
        detector = YOLODetector(
            model_name=self.model_name, conf=self.conf,
            iou=self.iou, imgsz=self.imgsz,
        )
        tracker = Tracker()
        trajectory_tracker = TrajectoryTracker(max_points=50)

        self.status.emit("Processing...")
        frame_count = 0
        prev_time = time.time()

        try:
            while not self._is_stopped():
                ret, frame = loader.read_frame()
                if not ret:
                    break

                # ── detect + track ────────────────────────────────────────
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)

                # ── draw ──────────────────────────────────────────────────
                processed_frame = frame.copy()
                active_ids = set()
                detection_count = len(detections)

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    active_ids.add(track_id)
                    ltrb = track.to_ltrb()
                    x1, y1 = int(ltrb[0]), int(ltrb[1])
                    x2, y2 = int(ltrb[2]), int(ltrb[3])

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    trajectory_tracker.update(track_id, (cx, cy))
                    trajectory_tracker.draw(processed_frame, track_id,
                                            color=(0, 255, 255), thickness=2)

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    det_class = (track.det_class
                                 if hasattr(track, 'det_class') and track.det_class
                                 else "Obj")
                    label = f"{det_class} {track_id}"
                    cv2.putText(processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                trajectory_tracker.cleanup(active_ids)

                # ── FPS ───────────────────────────────────────────────────
                now = time.time()
                elapsed = now - prev_time
                fps_val = 1.0 / elapsed if elapsed > 0 else 0.0
                prev_time = now
                frame_count += 1

                # ── save frame ────────────────────────────────────────────
                out.write(processed_frame)

                # ── emit QImage ───────────────────────────────────────────
                rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                # .copy() detaches from numpy buffer so it survives cross-thread
                qimg = QImage(rgb.data, w, h, bytes_per_line,
                              QImage.Format_RGB888).copy()

                stats = {
                    "fps": round(fps_val, 1),
                    "frame": frame_count,
                    "tracks": len(active_ids),
                    "detections": detection_count,
                    "model": self.model_name,
                }
                self.frame_ready.emit(qimg, stats)

        finally:
            # Always release resources, even on error / stop
            loader.release()
            out.release()

        self.finished.emit(output_path)
