import torch
import cv2
from ultralytics import YOLO

# COCO class IDs relevant to surveillance scenarios
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
_TARGET_CLASSES = {0, 1, 2, 3, 5, 7}

# Human-readable names for bounding-box labels
_CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


class YOLODetector:
    """
    YOLOv8-based object detector optimized for surveillance accuracy.

    [IPCV Syllabus Mapping - UNIT III: Object Detection, Recognition, and context]
      - Object detection: Deep Learning model locating bounded coordinates dynamically.
      - Instance recognition: Extrapolating specific features separating varying object types
        overlapping in contextual scenes.
      - Scene understanding: Agnostic NMS implementations isolating object boundaries locally.

    Key design choices (Phase 5 tuning):
      • Default model upgraded to yolov8s (small) — better accuracy than nano,
        still real-time on RTX 4060.
      • imgsz parameter exposed so inference can run at higher resolution
        (e.g. 1280) to catch small / distant objects.
      • Lower confidence threshold (0.25) keeps marginal detections that
        DeepSORT can smooth out over time.
      • Tighter NMS IoU (0.4) removes more duplicate boxes.
      • agnostic_nms enabled so overlapping boxes of different classes are
        also suppressed (person overlapping with bicycle, etc.).
      • classes filter passed directly to YOLO predict() so the model only
        outputs the 6 target classes — faster post-processing.
    """

    def __init__(
        self,
        model_name="yolov8s.pt",
        conf=0.25,
        iou=0.4,
        imgsz=1280,
    ):
        """
        Args:
            model_name : Weight file (downloaded on first run).
                         Recommended: yolov8s.pt (accuracy), yolov8n.pt (speed).
            conf       : Minimum confidence for a detection to survive.
            iou        : IoU threshold for NMS — lower = more aggressive suppression.
            imgsz      : Inference resolution. Higher = better for small objects
                         but slower.  640 / 1280 are common choices.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[YOLODetector] Using device: {self.device}")

        self.model = YOLO(model_name)
        self.model.to(self.device)

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        # Pre-build the list of COCO class IDs to pass to predict()
        self._target_class_list = sorted(_TARGET_CLASSES)

        print(f"[YOLODetector] Model: {model_name}  conf={conf}  iou={iou}  imgsz={imgsz}")

    def detect(self, frame):
        """
        Run YOLOv8 inference on a single BGR frame.

        Returns:
            detections : list of ([x, y, w, h], confidence, class_label)
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self._target_class_list,   # only detect target classes
            agnostic_nms=True,                  # cross-class NMS
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])

                # Extract xyxy → xywh
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1

                # Skip degenerate boxes
                if w < 5 or h < 5:
                    continue

                confidence = float(box.conf[0])
                class_label = _CLASS_NAMES.get(cls_id, "Object")

                detections.append(([int(x1), int(y1), int(w), int(h)],
                                   confidence, class_label))

        return detections
