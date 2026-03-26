import torch
from ultralytics import YOLO

# COCO class IDs we care about for surveillance
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
_TARGET_CLASSES = {0, 1, 2, 3, 5, 7}

# Readable names for display labels
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
    YOLOv8-based object detector.

    Wraps Ultralytics YOLOv8 to produce detections in the exact format
    expected by the DeepSORT tracker:
        ([left, top, width, height], confidence, class_label)

    Automatically uses CUDA if an NVIDIA GPU is available.
    """

    def __init__(self, model_name="yolov8n.pt", conf=0.35, iou=0.45):
        """
        Args:
            model_name: YOLOv8 model weight file (downloaded automatically on first run).
                        Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.
            conf:       Minimum confidence threshold for detections.
            iou:        IOU threshold for NMS (non-max suppression).
        """
        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[YOLODetector] Using device: {self.device}")

        # Load pretrained model
        self.model = YOLO(model_name)
        self.model.to(self.device)

        self.conf = conf
        self.iou = iou

    def detect(self, frame):
        """
        Run YOLOv8 inference on a single BGR frame.

        Args:
            frame: numpy array (H, W, 3) BGR image.

        Returns:
            detections: list of ([x, y, w, h], confidence, class_label)
                        ready for DeepSORT's update_tracks().
        """
        # Run inference (verbose=False suppresses per-frame logs)
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])

                # Keep only target classes (people + vehicles)
                if cls_id not in _TARGET_CLASSES:
                    continue

                # Extract xyxy and convert to xywh
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)

                confidence = float(box.conf[0])
                class_label = _CLASS_NAMES.get(cls_id, "Object")

                # DeepSORT format: ([left, top, width, height], confidence, class)
                detections.append(([x, y, w, h], confidence, class_label))

        return detections
