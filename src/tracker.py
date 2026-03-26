from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    """
    DeepSORT wrapper tuned for high-accuracy surveillance tracking (Phase 5).

    Tuning rationale
    ----------------
    max_age = 30        Longer window so tracks survive brief occlusions
                        (e.g. person walks behind a pole). Was 20 — too aggressive.
    n_init = 2          Confirm after just 2 hits so real objects appear quickly.
                        Was 3 — caused late ID assignment for fast-moving people.
    max_iou_distance    0.6 — slightly tighter spatial matching. Prevents a
                        detection from "jumping" to a far-away track.
    max_cosine_distance 0.2 — stricter appearance matching. Keeps distinct people
                        separate even when close together. Was 0.3.
    nn_budget = 150     More stored embeddings per track for better re-ID after
                        longer occlusions. Was 100.
    nms_max_overlap     0.8 — internal DeepSORT NMS to suppress duplicate tracks
                        that share the same detection.
    """

    def __init__(
        self,
        max_age=30,
        n_init=2,
        max_iou_distance=0.6,
        max_cosine_distance=0.2,
        nn_budget=150,
        nms_max_overlap=0.8,
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            nms_max_overlap=nms_max_overlap,
        )

    def update(self, detections, frame):
        """
        Feed new detections and the current frame to DeepSORT.

        Args:
            detections : list of ([x, y, w, h], confidence, class_label)
            frame      : BGR numpy array (used for Re-ID feature extraction)

        Returns:
            List of Track objects with .track_id, .to_ltrb(), .is_confirmed(), etc.
        """
        return self.tracker.update_tracks(detections, frame=frame)
