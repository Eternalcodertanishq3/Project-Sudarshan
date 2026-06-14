from pydantic import BaseModel
from typing import List

class SatellitePassModel(BaseModel):
    norad_id: int
    name: str
    sat_type: str
    is_threat: bool
    azimuth_deg: float
    elevation_deg: float
    range_km: float
    altitude_km: float
    is_surveillance_pass: bool
    surveillance_risk: float

class OrbitalBatch(BaseModel):
    timestamp: float
    active_passes: List[SatellitePassModel]
    total_overhead: int
