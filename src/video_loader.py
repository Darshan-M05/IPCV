import cv2
import os

class VideoLoader:
    def __init__(self, video_path):
        """
        Initializes the VideoLoader with the given video path.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Failed to open video file: {video_path}")
            
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def get_properties(self):
        """Returns video width, height, and fps."""
        return self.width, self.height, self.fps
        
    def read_frame(self):
        """Reads the next frame from the video."""
        ret, frame = self.cap.read()
        return ret, frame
        
    def release(self):
        """Releases the video capture object."""
        self.cap.release()
