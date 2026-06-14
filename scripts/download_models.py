import os
from pathlib import Path

def download_yolo():
    print("Downloading YOLOv10 model weights for offline operation...")
    try:
        from ultralytics import YOLO
        # Trigger an automatic download if it doesn't exist locally
        _ = YOLO('yolov10s.pt')
        print("Successfully downloaded yolov10s.pt")
    except ImportError:
        print("Please ensure ultralytics is installed: pip install ultralytics")
    except Exception as e:
        print(f"Failed to download YOLO model: {e}")

if __name__ == "__main__":
    download_yolo()
