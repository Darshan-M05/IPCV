import cv2
import numpy as np

class MotionDetector:
    def __init__(
        self,
        history=300,
        var_threshold=40,
        detect_shadows=False,         # Disable shadows: faster and avoids gray ghost blobs
        min_area=800,                  # Minimum pixel area for a valid detection
        max_area=80000,                # Maximum pixel area (filters out whole-sky/ground blobs)
        min_aspect_ratio=0.2,          # Width/height ratio floor  (remove thin horizontal noise)
        max_aspect_ratio=4.0,          # Width/height ratio ceiling (remove wide merged blobs)
        learning_rate=0.005,           # Slow background adaptation for surveillance scenes
    ):
        """
        Improved MOG2-based motion detector with:
        - Gaussian blur to smooth noise before subtraction
        - Dilation to fill small holes inside moving objects
        - Erosion to separate nearby/touching blobs
        - Area + aspect-ratio filtering to remove merged and spurious detections
        """
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area
        self.max_area = max_area
        self.min_ar = min_aspect_ratio
        self.max_ar = max_aspect_ratio
        self.learning_rate = learning_rate

        # Morphology kernels kept as instance variables (avoid re-creating every frame)
        self._open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame):
        """
        Apply Gaussian blur to the input frame before background subtraction.
        Reduces high-frequency sensor noise that causes spurious detections.
        """
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def _postprocess_mask(self, mask):
        """
        Clean the raw foreground mask with a pipeline of morphological operations:
          1. OPEN  (erode → dilate)  – removes small isolated noise pixels
          2. CLOSE (dilate → erode)  – fills holes inside object silhouettes
          3. Additional dilation     – gently expands blobs so thin limbs merge
                                       back into the body silhouette
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        mask = cv2.dilate(mask, self._dilate_kernel, iterations=1)
        return mask

    def _is_valid_detection(self, x, y, w, h):
        """
        Returns True when the bounding box passes all sanity checks:
          • Area within [min_area, max_area]
          • Aspect ratio (w/h) within [min_ar, max_ar]
        """
        area = w * h
        if area < self.min_area or area > self.max_area:
            return False
        ar = w / float(h) if h > 0 else 0
        if ar < self.min_ar or ar > self.max_ar:
            return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, frame):
        """
        Full detection pipeline.

        Returns:
            detections : list of ([x, y, w, h], confidence, class_label)
                         ready for DeepSORT's update_tracks().
            fg_mask    : cleaned binary foreground mask (for debug visualisation).
        """
        # Step 1 – denoise the frame
        blurred = self._preprocess(frame)

        # Step 2 – background subtraction with controlled learning rate
        raw_mask = self.back_sub.apply(blurred, learningRate=self.learning_rate)

        # Step 3 – threshold: keep only definite foreground (255), discard shadows (127)
        _, fg_mask = cv2.threshold(raw_mask, 254, 255, cv2.THRESH_BINARY)

        # Step 4 – clean the mask
        fg_mask = self._postprocess_mask(fg_mask)

        # Step 5 – find external contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Step 6 – build DeepSORT detection list with quality filtering
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if not self._is_valid_detection(x, y, w, h):
                continue

            # Confidence proportional to filled ratio (contour area / bbox area)
            # Penalises very sparse / ghostly contours
            filled_ratio = cv2.contourArea(cnt) / float(w * h)
            confidence = float(np.clip(filled_ratio, 0.3, 1.0))

            # DeepSORT expects: ([left, top, width, height], confidence, class)
            detections.append(([x, y, w, h], confidence, "object"))

        return detections, fg_mask
