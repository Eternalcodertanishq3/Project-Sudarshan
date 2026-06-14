"""
Pre-built demo scenarios — inject with one button click.
The judge sees the FULL RED ALERT sequence play out in real-time.
"""

import time

SCENARIOS = {

    "red_alert": {
        "scenario_name": "RED ALERT — DUAL DOMAIN INCURSION",
        "timestamp": None,  # Will be set to time.time() at injection

        "vision": {
            "active": True,
            "detections": [{
                "track_id": 1,
                "class_name": "UAV",
                "confidence": 0.85,
                "bbox_xywh": [640, 360, 80, 45],
                "speed_mps": 45.0,
                "heading_deg": 230.0
            }]
        },

        "orbital": {
            "active": True,
            "satellite_name": "YAOGAN-27 (Probable ISR)",
            "elevation_deg": 47.3,
            "azimuth_deg": 212.4,
            "range_km": 487.2,
            "is_surveillance_pass": True,
            "surveillance_risk": 0.89
        },

        "kinematic": {
            "occluded": False,
            "speed_mps": 45.0,
            "anomaly_detected": True,  # Evasive maneuver detected
            "predicted_vector": [850, 290]
        },

        "tactical": {
            "threat_probability": 0.97,
            "threat_level": "BLACK",
            "alert_state": "RED_ALERT",
            "message": "⚡ RED ALERT — UAV INCURSION + ISR SATELLITE OVERHEAD. P(THREAT)=97%",
            "recommended_action": "IMMEDIATE_INTERCEPTION_PROTOCOL"
        }
    },

    "occlusion_demo": {
        "scenario_name": "EKF OCCLUSION BRIDGING DEMO",
        "phases": [
            {"phase": "TRACKING",    "confidence": 0.91, "occluded": False, "frames": 60},
            {"phase": "OCCLUSION",   "confidence": 0.0,  "occluded": True,  "frames": 90},
            {"phase": "REACQUISITION","confidence": 0.87,"occluded": False, "frames": 30},
        ]
    },

    "orbital_pass": {
        "scenario_name": "SATELLITE SURVEILLANCE PASS",
        "passes": [
            {"name": "OBJECT-A (UNKNOWN PAYLOAD)", "elevation": 23.4, "risk": 0.72},
            {"name": "YAOGAN-14 (ISR SAT)",         "elevation": 51.2, "risk": 0.94},
            {"name": "STARLINK-1234 (CIVILIAN)",    "elevation": 67.8, "risk": 0.15},
        ]
    },

    "sea_domain": {
        "scenario_name": "MARITIME DOMAIN AWARENESS",
        "vessels": [
            {"class_name": "FAST_BOAT", "confidence": 0.79, "bearing_deg": 145, "range_km": 12.3},
            {"class_name": "FRIGATE",   "confidence": 0.93, "bearing_deg": 192, "range_km": 8.7},
        ],
        "threat_probability": 0.68,
        "alert_state": "WARNING"
    }
}
