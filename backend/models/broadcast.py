from pydantic import BaseModel
from typing import List, Optional
from models.detection import VisionFrame
from models.kinematic import KinematicBatch
from models.orbital import OrbitalBatch
from models.threat import ThreatAssessmentModel

class TacticalSummary(BaseModel):
    assessments: List[ThreatAssessmentModel]
    highest_threat_probability: float
    global_alert_state: str

class SudarshanPayload(BaseModel):
    timestamp: float
    frame_id: int
    system_state: str
    vision: VisionFrame
    kinematic: KinematicBatch
    orbital: OrbitalBatch
    tactical: TacticalSummary
