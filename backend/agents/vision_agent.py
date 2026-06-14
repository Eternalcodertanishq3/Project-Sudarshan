"""
VISION AGENT — YOLOv10 Perception Engine
Runs in isolated GPU process. 
Outputs: bounding boxes to shared queue at camera FPS.
"""

import cv2
import torch
import numpy as np
import multiprocessing as mp
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional
import time


@dataclass
class Detection:
    """Single target detection from YOLO inference."""
    bbox_xyxy: List[float]          # [x1, y1, x2, y2] in pixels
    bbox_xywh: List[float]          # [cx, cy, w, h] in pixels
    confidence: float               # 0.0 to 1.0
    class_id: int
    class_name: str
    track_id: Optional[int] = None  # Set by tracker
    timestamp: float = 0.0

    def centroid(self) -> tuple:
        """Returns (cx, cy) centroid of bounding box."""
        return (
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2
        )


class VisionAgent:
    """
    YOLOv10 perception engine.
    IoU threshold for NMS is the Equation (1) from the spec document.
    
    IoU = Area_of_Overlap / Area_of_Union
    If IoU > 0.45 between two boxes → discard lower-confidence box.
    This is handled internally by the YOLO model but we can inspect it.
    """

    # Map COCO class IDs to military-relevant threat categories
    # The Hackathon Trick: Re-map default COCO classes to military outputs
    THREAT_MAP = {
        # COCO class → (military_label, threat_level, domain)
        2:  ("TACTICAL_VEHICLE",  "MEDIUM", "LAND"),    # Car
        4:  ("UNIDENTIFIED_UAV",  "HIGH",   "AIR"),     # Airplane
        5:  ("TACTICAL_VEHICLE",  "MEDIUM", "LAND"),    # Bus
        7:  ("TACTICAL_VEHICLE",  "HIGH",   "LAND"),    # Truck
        8:  ("HOSTILE_VESSEL",    "HIGH",   "SEA"),     # Boat
        
        # Explicitly dropping civilian noise: person(0), bird(14), kite(38) will be ignored
    }

    def __init__(
        self,
        model_path: str = "yolov10s.pt",
        conf_threshold: float = 0.30,        # Lowered to catch uncertain military hardware
        iou_threshold: float = 0.45,         # NMS IoU threshold from Eq.(1)
        input_size: int = 640,
        use_gpu: bool = True
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Device selection
        self.device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
        print(f"[VisionAgent] Loading YOLOv10 on {self.device.upper()}")

        # Load model — downloads automatically on first run
        self.model = YOLO(model_path)
        self.model.to(self.device)

        print(f"[VisionAgent] Model loaded. Input tensor: (1×3×{input_size}×{input_size})")

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert BGR frame to normalized RGB tensor.
        Shape: (1, 3, 640, 640) — the input tensor from the spec doc.
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        # BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # HWC → CHW, normalize 0-255 → 0-1
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        return tensor.unsqueeze(0).to(self.device)

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """
        Full inference pass:
        1. Preprocess → tensor
        2. Forward pass through YOLOv10 conv layers
        3. NMS filters overlapping boxes by IoU > 0.45
        4. Returns cleaned detections
        """
        h_orig, w_orig = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,     # This is the NMS IoU threshold
            imgsz=self.input_size,
            verbose=False,
            stream=False
        )

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if class_id not in self.THREAT_MAP:
                    continue

                military_label, threat_level, domain = self.THREAT_MAP[class_id]
                conf = float(box.conf[0].item())

                # Scale bounding box back to original resolution
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                scale_x = w_orig / self.input_size
                scale_y = h_orig / self.input_size
                x1 *= scale_x; x2 *= scale_x
                y1 *= scale_y; y2 *= scale_y
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                det = Detection(
                    bbox_xyxy=[x1, y1, x2, y2],
                    bbox_xywh=[cx, cy, w, h],
                    confidence=conf,
                    class_id=class_id,
                    class_name=military_label,
                    timestamp=time.time()
                )
                detections.append(det)

        return detections

    def run(self, frame_queue: mp.Queue, detection_queue: mp.Queue, stop_event: mp.Event):
        """
        Main loop — Process A.
        Reads frames from camera or simulator, runs inference, pushes to queue.
        """
        print("[VisionAgent] Process A running...")
        while not stop_event.is_set():
            try:
                frame_data = frame_queue.get(timeout=0.1)
                detections = self.infer(frame_data['frame'])

                # Push detections to kinematic agent
                detection_queue.put({
                    'timestamp': time.time(),
                    'frame_id': frame_data.get('frame_id', 0),
                    'detections': [
                        {
                            'bbox_xyxy': d.bbox_xyxy,
                            'bbox_xywh': d.bbox_xywh,
                            'confidence': d.confidence,
                            'class_name': d.class_name,
                            'timestamp': d.timestamp
                        }
                        for d in detections
                    ],
                    'frame_width': frame_data['frame'].shape[1],
                    'frame_height': frame_data['frame'].shape[0],
                })
            except Exception:
                pass
