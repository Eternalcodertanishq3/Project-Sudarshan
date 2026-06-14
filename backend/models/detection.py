from pydantic import BaseModel
from typing import List, Optional

class DetectionBase(BaseModel):
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    confidence: float
    class_name: str
    timestamp: float

class VisionFrame(BaseModel):
    frame_id: int
    timestamp: float
    detections: List[DetectionBase]
    frame_width: int
    frame_height: int
