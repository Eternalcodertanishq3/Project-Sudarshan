import cv2
import numpy as np
import urllib.request
import os
from pathlib import Path

# URLs for real-world images (Wikimedia Commons - Public Domain / CC)
IMAGE_URLS = {
    "air_drone.jpg": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/kite.jpg",
    "sea_ship.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
    "land_truck.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
}

def download_images():
    Path("tests/data").mkdir(parents=True, exist_ok=True)
    for filename, url in IMAGE_URLS.items():
        filepath = f"tests/data/{filename}"
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            # Use a User-Agent to avoid 403 Forbidden
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Saved {filepath}")

def create_panning_video(image_path, output_path, duration=3, fps=30):
    """Creates a video by slowly panning across a real image to simulate motion for the EKF."""
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
        
    h, w, _ = img.shape
    # We will crop a window and slide it
    window_w = int(w * 0.7)
    window_h = int(h * 0.7)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (window_w, window_h))
    
    total_frames = duration * fps
    for i in range(total_frames):
        # Calculate sliding offset
        progress = i / float(total_frames)
        start_x = int(progress * (w - window_w))
        start_y = int(progress * (h - window_h))
        
        # Crop
        frame = img[start_y:start_y+window_h, start_x:start_x+window_w]
        
        # Optional: Add slight camera shake or noise to simulate real sensor
        # noise = np.random.normal(0, 2, frame.shape).astype(np.uint8)
        # frame = cv2.add(frame, noise)
        
        out.write(frame)
        
    out.release()
    print(f"Generated simulated motion video: {output_path}")

if __name__ == "__main__":
    print("Preparing Real-World Test Datasets...")
    download_images()
    
    create_panning_video("tests/data/air_drone.jpg", "tests/data/air_domain.mp4")
    create_panning_video("tests/data/sea_ship.jpg", "tests/data/sea_domain.mp4")
    create_panning_video("tests/data/land_truck.jpg", "tests/data/land_domain.mp4")
    print("Datasets ready.")
