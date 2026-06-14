import numpy as np

class DronePathGenerator:
    """Generates synthetic trajectories for the EKF to track."""
    def __init__(self, start_x=0, start_y=100, speed_x=5, speed_y=0):
        self.x = start_x
        self.y = start_y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.tick = 0

    def step(self):
        self.tick += 1
        self.x += self.speed_x
        self.y += self.speed_y + np.sin(self.tick * 0.1) * 5
        
        # Predict occlusion from tick 50 to 90
        is_occluded = 50 < self.tick < 90
        
        return self.x, self.y, is_occluded
