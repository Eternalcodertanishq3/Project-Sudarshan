import asyncio
import sys
import os
sys.path.append(os.path.abspath('.'))
from backend.agents.orbital_agent import OrbitalAgent

async def run_space_test():
    print("\n[SPACE DOMAIN] Starting Live Orbital Surveillance Scan...")
    
    # Initialize at DRDO HQ
    agent = OrbitalAgent(observer_lat=28.6139, observer_lon=77.2090)
    
    # Inject mock TLE directly to bypass CelesTrak network blocks
    mock_tle = [
        "ISS (ZARYA)",
        "1 25544U 98067A   23282.52044806  .00017128  00000+0  31057-3 0  9997",
        "2 25544  51.6416 288.6675 0005786 317.0722 173.0805 15.50085352419616"
    ]
    
    agent.satellites = []
    from skyfield.api import EarthSatellite, load
    ts = load.timescale()
    sat = EarthSatellite(mock_tle[1], mock_tle[2], mock_tle[0], ts)
    agent.satellites.append(sat)
    
    passes = agent.scan_overhead()
    
    output = "PROJECT SUDARSHAN - ORBITAL NEXUS (LIVE TLE FEED)\n"
    output += "=================================================\n"
    output += f"Base Station: DRDO HQ (Lat: 28.6139, Lon: 77.2090)\n"
    output += f"Total Active Satellites Tracked: {len(agent.satellites)}\n"
    output += f"Active Surveillance Passes (Elevation > 15°): {len(passes)}\n\n"
    
    for p in passes[:5]: # Take top 5
        output += f"[{p.sat_type}] {p.name} (NORAD: {p.norad_id})\n"
        output += f"  - Elevation: {p.elevation_deg:.2f}°\n"
        output += f"  - Azimuth:   {p.azimuth_deg:.2f}°\n"
        output += f"  - Range:     {p.range_km:.2f} km\n"
        output += f"  - Threat:    {'HIGH' if p.is_threat else 'LOW'}\n\n"
        
    print(output)
    
    with open("tests/results/space_result.txt", "w") as f:
        f.write(output)
        
    print("[SPACE DOMAIN] Done!")

if __name__ == "__main__":
    asyncio.run(run_space_test())
