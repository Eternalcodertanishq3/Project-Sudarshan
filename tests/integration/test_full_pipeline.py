import pytest
import time
from backend.agents.kinematic_agent import MultiTargetTracker
from backend.agents.tactical_agent import BayesianFusionEngine, ThreatEvidence

@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """
    Simulates the entire Quad-Domain C4ISR pipeline mathematically.
    Injects a mock YOLO detection and asserts a Threat Payload is generated.
    """
    tracker = MultiTargetTracker()
    tactical_engine = BayesianFusionEngine()
    
    # 1. Simulate Vision Detection
    vision_payload = [{
        'bbox_xyxy': [100, 100, 150, 150],
        'bbox_xywh': [125, 125, 50, 50],
        'confidence': 0.88,
        'class_name': 'UAV'
    }]
    
    # 2. Kinematic EKF Update
    tracks = tracker.update(vision_payload)
    assert len(tracks) == 1
    
    # 3. Tactical Fusion Update
    # Form the evidence
    track = tracks[0]
    evidence = ThreatEvidence(
        vision_confidence=0.88,
        detected_class='UAV',
        track_speed_mps=track.speed,
        is_occluded=track.is_occluded,
        satellite_elevation_deg=20.0
    )
    
    assessment = tactical_engine.assess_threat(evidence=evidence, track_id=track.track_id)
    
    # Threat probability should be calculated
    assert assessment.threat_probability > 0.0
    assert assessment.threat_level in ["GREEN", "YELLOW", "ORANGE", "RED", "BLACK"]
    print("End-to-End pipeline verified successfully.")
