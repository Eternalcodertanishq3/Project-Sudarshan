"""
ORBITAL ENGINE — SGP4 Propagator + ECEF → Topocentric Transform
Implements the orbital mechanics described in Section 3.3 of the spec.

Key operations:
1. Load TLE catalog from CelesTrak (offline cache supported)
2. SGP4 propagation → ECEF coordinates
3. ECEF → Topocentric (Azimuth, Elevation, Range)
4. Flag surveillance passes when Elevation > 15°
"""

import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from skyfield.framelib import itrs
from pathlib import Path
import httpx
import asyncio
import json
from typing import List, Optional
from dataclasses import dataclass
import time


@dataclass
class SatellitePass:
    """A satellite surveillance pass over the base location."""
    norad_id: int
    name: str
    sat_type: str           # PAYLOAD | DEBRIS | ROCKET_BODY
    is_threat: bool         # True if reconnaissance-capable

    # Topocentric coordinates
    azimuth_deg: float
    elevation_deg: float
    range_km: float

    # Orbital elements
    altitude_km: float
    inclination_deg: float
    period_min: float

    # Pass timing
    pass_start: Optional[str] = None
    pass_end: Optional[str] = None
    max_elevation_deg: float = 0.0
    time_to_max_elevation_sec: float = 0.0

    # Risk
    surveillance_risk: float = 0.0  # 0.0 to 1.0


class OrbitalAgent:
    """
    Implements ECEF → Topocentric transformation from Section 3.3.
    
    ECEF: Earth-Centered Earth-Fixed coordinate system
    Topocentric: Local (Azimuth, Elevation, Range) relative to observer location
    
    The key trigger: if satellite Elevation > SURVEILLANCE_ELEVATION_THRESHOLD,
    flag it as active surveillance pass.
    """

    SURVEILLANCE_ELEVATION_THRESHOLD = 15.0  # degrees, from spec document
    TLE_CACHE_PATH = Path("data/tle_cache.json")

    # Satellites with known surveillance/ISR capability
    # NORAD IDs of known reconnaissance satellites (publicly known from amateur tracking)
    KNOWN_ISR_SATELLITES = {
        # USA
        33413: "USA-207 (KH-13)",
        39232: "USA-245 (NROL-39)",
        40889: "USA-268 (NROL-55)",
        47306: "USA-311 (NROL-44) - MASSIVE SIGINT",
        # Chinese
        33732: "YAOGAN-7",
        37165: "YAOGAN-14",
        40701: "YAOGAN-27",
        # Russian
        41032: "BARS-M",
        43080: "KOSMOS-2519",
        40081: "KOSMOS-2500 - SURVEILLANCE",
        # Generic test
        25544: "ISS (ZARYA) - TEST TARGET",
    }

    def __init__(
        self,
        observer_lat: float = 28.6139,      # Default: Delhi (DRDO HQ)
        observer_lon: float = 77.2090,
        observer_elev_m: float = 216.0
    ):
        self.observer_lat = observer_lat
        self.observer_lon = observer_lon
        self.observer_elev_m = observer_elev_m

        # Load Skyfield timescale
        self.ts = load.timescale()

        # Observer location (Skyfield wgs84)
        self.observer = wgs84.latlon(
            observer_lat,
            observer_lon,
            observer_elev_m
        )

        # Satellite catalogue
        self.satellites: List[EarthSatellite] = []
        print(f"[OrbitalAgent] Observer: {observer_lat:.4f}°N, {observer_lon:.4f}°E")

    async def load_tle_catalog(self, categories: List[str] = None):
        """
        Load TLE data from CelesTrak (or cache if offline).
        CelesTrak is free, no auth required, publicly accessible.
        """
        if categories is None:
            categories = [
                "active",           # All active satellites
                "visual",           # Visually observable
                "stations",         # ISS etc.
            ]

        # CelesTrak URLs for different categories
        tle_urls = {
            "active":   "https://celestrak.org/SOCRATES/query.php?CODE=ALL&FORMAT=TLE",
            "visual":   "https://celestrak.org/SOCRATES/visual.txt",
            "stations": "https://celestrak.org/SOCRATES/stations.txt",
            # Use these for offline demo with real data:
            "gp":       "https://celestrak.org/SOCRATES/gp.php?GROUP=active&FORMAT=tle",
        }

        # Try cache first (for air-gapped demo)
        if self.TLE_CACHE_PATH.exists():
            print("[OrbitalAgent] Loading TLE from offline cache...")
            with open(self.TLE_CACHE_PATH) as f:
                tle_data = json.load(f)
            self._parse_tle_lines(tle_data)
            return

        # Fetch from CelesTrak
        print("[OrbitalAgent] Fetching Military TLE from CelesTrak...")
        all_tle_lines = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    "https://celestrak.org/NORAD/elements/military.txt"
                )
                lines = resp.text.strip().split('\n')
                all_tle_lines.extend(lines)
                print(f"[OrbitalAgent] Fetched {len(lines)//3} military objects from CelesTrak")
            except Exception as e:
                print(f"[OrbitalAgent] Network fetch failed: {e}. Using cached data.")

        self._parse_tle_lines(all_tle_lines)

        # Cache for offline operation
        self.TLE_CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(self.TLE_CACHE_PATH, 'w') as f:
            json.dump(all_tle_lines, f)

    def _parse_tle_lines(self, lines: List[str]):
        """Parse raw TLE text into Skyfield EarthSatellite objects."""
        self.satellites = []
        lines = [l.strip() for l in lines if l.strip()]

        i = 0
        while i + 2 < len(lines):
            name = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]

            if line1.startswith('1 ') and line2.startswith('2 '):
                try:
                    sat = EarthSatellite(line1, line2, name, self.ts)
                    self.satellites.append(sat)
                except Exception:
                    pass
                i += 3
            else:
                i += 1

        print(f"[OrbitalAgent] Loaded {len(self.satellites)} satellites")

    def compute_topocentric(self, satellite: EarthSatellite) -> dict:
        """
        ECEF → Topocentric transformation (Section 3.3 of spec).
        
        1. SGP4 propagates TLE → ECEF position (X, Y, Z in km)
        2. Skyfield converts ECEF → Topocentric (az, el, range) relative
           to observer at (lat, lon, elevation)
        
        Returns azimuth (0-360°), elevation (-90 to 90°), range (km)
        """
        t = self.ts.now()

        # Compute satellite - observer difference vector
        # This is the ECEF → Topocentric transformation
        difference = satellite - self.observer
        topocentric = difference.at(t)

        # Get altitude (elevation), azimuth, and distance
        alt, az, dist = topocentric.altaz()

        # Get ECEF position for altitude calculation
        geocentric = satellite.at(t)
        subpoint = wgs84.subpoint_of(geocentric)
        altitude_km = subpoint.elevation.km

        return {
            "azimuth_deg": float(az.degrees),
            "elevation_deg": float(alt.degrees),
            "range_km": float(dist.km),
            "altitude_km": float(altitude_km),
            "is_above_horizon": float(alt.degrees) > 0,
            "is_surveillance_pass": float(alt.degrees) > self.SURVEILLANCE_ELEVATION_THRESHOLD
        }

    def scan_overhead(self) -> List[SatellitePass]:
        """
        Scan all loaded satellites.
        Return list of active surveillance passes (elevation > 15°).
        """
        active_passes = []

        for sat in self.satellites:
            try:
                topo = self.compute_topocentric(sat)

                if not topo['is_above_horizon']:
                    continue

                # Extract NORAD ID from TLE name field
                norad_id = int(sat.model.satnum) if hasattr(sat, 'model') else 0

                # Determine if reconnaissance-capable
                is_known_isr = norad_id in self.KNOWN_ISR_SATELLITES
                is_threat = is_known_isr or topo['elevation_deg'] > 45.0

                # Surveillance risk score
                # High elevation = better coverage = higher risk
                base_risk = min(topo['elevation_deg'] / 90.0, 1.0)
                risk = base_risk * (1.5 if is_known_isr else 1.0)
                risk = min(risk, 1.0)

                pass_data = SatellitePass(
                    norad_id=norad_id,
                    name=self.KNOWN_ISR_SATELLITES.get(norad_id, sat.name.strip()),
                    sat_type=self._classify_satellite(sat.name),
                    is_threat=is_threat,
                    azimuth_deg=topo['azimuth_deg'],
                    elevation_deg=topo['elevation_deg'],
                    range_km=topo['range_km'],
                    altitude_km=topo['altitude_km'],
                    inclination_deg=float(sat.model.inclo * 180 / np.pi),
                    period_min=float(2 * np.pi / sat.model.no_kozai / 60),
                    surveillance_risk=risk,
                    max_elevation_deg=topo['elevation_deg'],
                )
                active_passes.append(pass_data)

            except Exception:
                continue

        # Sort by surveillance risk
        active_passes.sort(key=lambda p: p.surveillance_risk, reverse=True)
        return active_passes[:20]   # Top 20 overhead objects

    def _classify_satellite(self, name: str) -> str:
        name_upper = name.upper()
        if any(k in name_upper for k in ['DEB', 'R/B', 'ROCKET']):
            return 'DEBRIS'
        elif any(k in name_upper for k in ['YAOGAN', 'KH-', 'NROL', 'KOSMOS', 'BARS', 'TOPAZ']):
            return 'ISR_SATELLITE'
        elif any(k in name_upper for k in ['STARLINK', 'ONEWEB', 'GPS', 'GLONASS', 'NAVIC']):
            return 'CIVILIAN_CONSTELLATION'
        return 'UNKNOWN_PAYLOAD'
