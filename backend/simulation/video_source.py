import cv2

class VideoSource:
    """Unified video source handling webcam or simulator."""
    def __init__(self, source_id=0, use_simulator=False):
        self.use_simulator = use_simulator
        if self.use_simulator:
            from simulation.camera_simulator import CameraSimulator
            self.cap = CameraSimulator()
        else:
            self.cap = cv2.VideoCapture(source_id)

    def read(self):
        return self.cap.read()
        
    def release(self):
        if not self.use_simulator:
            self.cap.release()
