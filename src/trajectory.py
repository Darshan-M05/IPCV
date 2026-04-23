import collections
import cv2

class TrajectoryTracker:
    def __init__(self, max_points=50):
        """
        Initializes the Trajectory Tracker.

        [IPCV Syllabus Mapping - UNIT V: Object Measurement, Pose Estimation and Understanding]
          - Object Measurement: Counting continuous positional locations to map speed vectors and 
            trajectory lengths.
          - Image Understanding: Connecting spatial dots to visually output analytical path trails.

        max_points: Max number of history points to keep for a trajectory.
        """
        # Dictionary to store trajectories. Key is track_id, value is a deque of points.
        self.trajectories = collections.defaultdict(lambda: collections.deque(maxlen=max_points))
        self.max_points = max_points

    def update(self, track_id, center_point):
        """
        Adds a new center point to the trajectory of track_id.
        """
        self.trajectories[track_id].append(center_point)
        
    def draw(self, frame, track_id, color=(0, 255, 255), thickness=2):
        """
        Draws the trajectory lines for a specific track_id on the given frame.
        """
        if track_id not in self.trajectories:
            return
            
        points = list(self.trajectories[track_id])
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            
            # Draw line between previous point and current point
            cv2.line(frame, points[i - 1], points[i], color, thickness)
            
    def cleanup(self, active_track_ids):
        """
        Removes trajectories for objects that have disappeared (not in active_track_ids).
        Helps keep memory usage efficient.
        """
        # Find track IDs to remove
        ids_to_remove = [tid for tid in self.trajectories.keys() if tid not in active_track_ids]
        for tid in ids_to_remove:
            self.trajectories.pop(tid, None)
