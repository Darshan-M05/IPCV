from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(
        self,
        max_age=20,              # Frames to keep a track alive without a detection match
        n_init=3,                # Frames a track must be seen before it is confirmed
        max_iou_distance=0.7,    # IOU threshold for matching detections to tracks
        max_cosine_distance=0.3, # Re-ID embedding similarity threshold (lower = stricter)
        nn_budget=100,           # Max stored embeddings per track
    ):
        """
        DeepSORT tracker tuned for surveillance scenes with multiple close objects.

        Key tuning rationale:
          max_age=20          – Shorter deletion window reduces ghost tracks after occlusion.
          n_init=3            – Require 3 consecutive hits before confirming a new track,
                                 avoids noise-spawned transient IDs.
          max_iou_distance    – Controls spatial matching aggressiveness.
          max_cosine_distance – Stricter re-ID distance separates look-alike people.
          nn_budget=100       – Limits memory while retaining enough history for re-ID.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

    def update(self, detections, frame):
        """
        Updates the tracker with the latest frame detections.

        Args:
            detections: list of ([x, y, w, h], confidence, class_label)
            frame:      current BGR frame (used for appearance feature extraction)

        Returns:
            List of active Track objects.
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks
