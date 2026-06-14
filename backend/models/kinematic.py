from pydantic import BaseModel
from typing import List, Optional, Tuple

class KinematicStateOutput(BaseModel):
    track_id: int
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    speed_mps: float
    heading_deg: float
    is_occluded: bool
    frames_lost: int
    uncertainty_radius_px: float
    predicted_reacquisition: Optional[Tuple[float, float]] = None

class KinematicBatch(BaseModel):
    timestamp: float
    tracks: List[KinematicStateOutput]
