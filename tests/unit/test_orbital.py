import pytest
from backend.agents.orbital_agent import OrbitalAgent
import asyncio

@pytest.mark.asyncio
async def test_orbital_propagation():
    """Test SGP4 propagation and ECEF to Topocentric conversion."""
    agent = OrbitalAgent(observer_lat=28.6139, observer_lon=77.2090)
    
    # Load a hardcoded TLE for a known satellite to avoid network dependency in tests
    mock_tle = [
        "ISS (ZARYA)",
        "1 25544U 98067A   23272.50000000  .00015504  00000-0  28254-3 0  9997",
        "2 25544  51.6416 114.7371 0004990 100.8665 240.8540 15.50043878418042"
    ]
    agent._parse_tle_lines(mock_tle)
    assert len(agent.satellites) == 1
    
    # Force the time to a known state or just test the math executes without error
    passes = agent.scan_overhead()
    
    # Test the core ECEF to Topocentric math calculation directly
    sat = agent.satellites[0]
    topo = agent.compute_topocentric(sat)
    
    assert 'azimuth_deg' in topo
    assert 'elevation_deg' in topo
    assert 'range_km' in topo
    
    # Distance to ISS should always be > 100km (orbit is ~400km)
    assert topo['range_km'] > 100.0
    print(f"ISS Current Range: {topo['range_km']:.2f} km")
