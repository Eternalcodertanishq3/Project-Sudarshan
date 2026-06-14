import pytest
import numpy as np
from backend.agents.kinematic_agent import ExtendedKalmanFilter, MultiTargetTracker

def test_ekf_prediction_accuracy():
    """Test that the EKF correctly predicts motion during an occlusion."""
    tracker = MultiTargetTracker(dt=0.1)
    
    # Simulate a drone moving diagonally at 10 pixels per frame
    detections = []
    for i in range(10):
        detections.append([{
            'bbox_xywh': [100 + i*10, 100 + i*10, 20, 20],
            'confidence': 0.9,
            'class_name': 'UAV'
        }])
        
    # Feed 10 frames of clear visibility
    for det in detections:
        tracks = tracker.update(det)
        
    assert len(tracks) == 1
    track = tracks[0]
    
    # After 10 frames, position should be near (190, 190) and velocity near (100, 100) per second (since dt=0.1)
    assert np.isclose(track.position[0], 190.0, atol=5.0)
    assert np.isclose(track.position[1], 190.0, atol=5.0)
    assert track.speed > 50.0  # Velocity is established
    
    # Simulate 5 frames of OCCLUSION (empty detections)
    for _ in range(5):
        tracks = tracker.update([])
        
    assert len(tracks) == 1
    track = tracks[0]
    assert track.is_occluded is True
    assert track.frames_lost == 5
    
    # The EKF should have predicted the position to continue diagonally
    # 5 frames * 10 pixels/frame = 50 pixels further
    assert np.isclose(track.position[0], 240.0, atol=15.0)
    assert np.isclose(track.position[1], 240.0, atol=15.0)
    print(f"Predicted Position during occlusion: {track.position}")
