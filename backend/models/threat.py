from pydantic import BaseModel
from typing import Dict, Optional

class ThreatEvidenceModel(BaseModel):
    vision: float
    orbital_elevation: Optional[float] = None
    kinematic_speed: Optional[float] = None

class ThreatAssessmentModel(BaseModel):
    threat_id: str
    threat_probability: float
    threat_level: str
    alert_state: str
    recommended_action: str
    alert_message: str
    evidence_breakdown: ThreatEvidenceModel
