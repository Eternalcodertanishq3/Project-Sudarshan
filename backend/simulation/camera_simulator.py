import cv2
import numpy as np

class CameraSimulator:
    """Mock camera that outputs synthetic frames for testing."""
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0

    def read(self):
        self.frame_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Synthetic horizon
        frame[self.height//2:, :] = (50, 50, 50)
        
        # Draw moving target (UAV)
        cx = int((self.frame_count * 5) % self.width)
        cy = self.height // 4 + int(np.sin(self.frame_count * 0.1) * 20)
        
        # Occlusion block
        if cx > 200 and cx < 400:
            pass # Target is behind a mountain
        else:
            cv2.rectangle(frame, (cx-10, cy-5), (cx+10, cy+5), (0, 0, 255), -1)
            
        return True, frame
