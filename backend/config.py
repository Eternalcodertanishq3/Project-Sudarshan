import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # System
    TARGET_FPS = 30
    BROADCAST_HZ = 60
    
    # EKF & Vision
    YOLO_MODEL_PATH = "yolov10s.pt"
    VISION_CONF_THRESHOLD = 0.45
    VISION_IOU_THRESHOLD = 0.45
    MAX_OCCLUSION_FRAMES = 90
    
    # Orbital
    SURVEILLANCE_ELEVATION_THRESHOLD = 15.0
    TLE_CACHE_PATH = BASE_DIR / "backend" / "data" / "tle_cache.txt"
    
    # Tactical Fusion
    BAYESIAN_PRIOR = 0.1
    ALERT_THRESHOLDS = {
        "LOW": 0.2,
        "MEDIUM": 0.5,
        "HIGH": 0.8,
        "BLACK": 0.95
    }

config = Config()
