import pytest
from backend.agents.tactical_agent import BayesianFusionEngine

def test_bayesian_update():
    """Test the Bayesian sensor fusion math."""
    engine = BayesianFusionEngine()
    
    # Vision agent reports a drone (high confidence)
    likelihood_vis = engine.likelihood_vision(confidence=0.85)
    
    # Kinematic agent reports high velocity matching a drone
    likelihood_kin = engine.likelihood_kinematic(speed_mps=25.0, is_anomalous=False)
    
    # Orbital agent confirms no satellite overhead
    likelihood_orb = engine.likelihood_orbital(elevation_deg=0.0)
    
    # Perform Sequential Bayes Update
    evidence_list = [likelihood_vis, likelihood_kin, likelihood_orb]
    
    posterior = engine.sequential_bayes_update(evidence_list, prior=0.05)
    
    print(f"Final fused threat probability: {posterior:.4f}")
    # Prior was 0.05. Vision and Kinematic should push it very high.
    assert posterior > 0.40
