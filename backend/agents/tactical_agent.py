"""
TACTICAL AGENT — Bayesian Sensor Fusion + Threat Assessment
Implements Equation (4) from the spec document.

P(T|Ev, Eo) = P(Ev, Eo|T) * P(T) / P(Ev, Eo)

Fuses weak signals from Vision + Kinematic + Orbital + RF (simulated)
into a single high-confidence threat probability.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


@dataclass
class ThreatEvidence:
    """Evidence bundle from all sensor domains."""

    # Vision domain (from Vision Agent)
    vision_confidence: float = 0.0      # Ev in spec — YOLO confidence
    detected_class: str = "UNKNOWN"

    # Kinematic domain (from Kinematic Agent)
    track_speed_mps: float = 0.0        # Speed in m/s (normalized from pixels)
    track_heading_deg: float = 0.0
    kinematic_anomaly: bool = False     # Abnormal trajectory detected
    is_occluded: bool = False

    # Orbital domain (from Orbital Agent)
    satellite_elevation_deg: float = 0.0   # Eo in spec
    satellite_overhead: bool = False        # Is surveillance sat above 15°?
    surveillance_risk: float = 0.0

    # RF domain (simulated for demo)
    rf_signature_detected: bool = False
    rf_frequency_mhz: float = 0.0
    rf_signal_strength_dbm: float = -120.0

    # Contextual
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThreatAssessment:
    """Output of Bayesian fusion for a single threat source."""
    threat_id: str
    threat_probability: float       # P(T|all evidence)
    threat_level: str               # GREEN/YELLOW/ORANGE/RED/BLACK
    alert_state: str                # NORMAL/WATCH/WARNING/ALERT/RED_ALERT
    confidence: float

    # Evidence contributions
    evidence_breakdown: Dict[str, float] = field(default_factory=dict)

    # Command output
    threat_coordinates: Optional[Tuple[float, float]] = None
    recommended_action: str = "CONTINUE_MONITORING"
    alert_message: str = ""
    timestamp: float = field(default_factory=time.time)

    # Auto-computed
    @property
    def is_red_alert(self) -> bool:
        return self.threat_probability >= 0.95


class BayesianFusionEngine:
    """
    Implements the Bayesian sensor fusion from Section 3.4 and Eq. (4).
    
    P(T|Ev, Eo) = P(Ev,Eo|T) * P(T)
                  ─────────────────────
                       P(Ev, Eo)
    
    Extended to N evidence sources with conditional independence assumption:
    P(T|E1,...,En) ∝ P(T) * ∏ P(Ei|T)
    
    This is the "sequential Bayes update" form — each evidence updates
    the running posterior one at a time.
    """

    # Prior probability of a real threat at any given moment
    # In peacetime monitoring: prior is low
    PRIOR_THREAT = 0.05  # 5% base rate

    # Threat level thresholds
    THRESHOLDS = {
        "GREEN":    (0.00, 0.30),   # Normal operations
        "YELLOW":   (0.30, 0.55),   # Watch — elevated activity
        "ORANGE":   (0.55, 0.75),   # Warning — probable threat
        "RED":      (0.75, 0.90),   # Alert — highly probable threat
        "BLACK":    (0.90, 1.00),   # RED ALERT — near-certain threat
    }

    # Likelihood functions P(evidence|threat) for each evidence type
    # These encode domain knowledge about what evidence looks like given a threat
    @staticmethod
    def likelihood_vision(confidence: float) -> Tuple[float, float]:
        """
        Returns (P(E|threat), P(E|no_threat))
        High YOLO confidence → more likely a real threat.
        """
        p_given_threat = 0.3 + 0.65 * confidence   # 0.30 to 0.95
        p_given_no_threat = 0.05 + 0.10 * confidence  # 0.05 to 0.15
        return p_given_threat, p_given_no_threat

    @staticmethod
    def likelihood_orbital(elevation_deg: float) -> Tuple[float, float]:
        """
        Returns (P(E|threat), P(E|no_threat))
        Orbital elevation > 15° significantly raises threat posterior.
        A spy satellite overhead during an incursion = not coincidence.
        """
        if elevation_deg < 5.0:
            return 0.10, 0.08   # Below horizon — background
        elif elevation_deg < 15.0:
            return 0.20, 0.10
        elif elevation_deg < 45.0:
            return 0.70, 0.15   # Overhead pass
        else:
            return 0.90, 0.05   # High overhead = likely ISR
        # Note: These numbers represent how much more likely
        # a satellite overhead is during a threat event vs. random

    @staticmethod
    def likelihood_kinematic(speed_mps: float, is_anomalous: bool) -> Tuple[float, float]:
        """
        Kinematic behavior consistent with threat profiles.
        Military UAVs: 15-80 m/s, erratic patterns
        Birds: <15 m/s, smooth paths
        Commercial drones: 5-25 m/s, smooth paths
        """
        # Speed-based likelihood
        if speed_mps < 5:
            speed_lh = (0.10, 0.15)    # Slow = probably not a threat
        elif speed_mps < 25:
            speed_lh = (0.40, 0.35)    # Medium — ambiguous
        elif speed_mps < 80:
            speed_lh = (0.75, 0.10)    # Fast — likely military
        else:
            speed_lh = (0.60, 0.05)    # Very fast — possibly missile-like

        # Anomalous trajectory (evasive maneuvers)
        if is_anomalous:
            return (speed_lh[0] * 1.3, speed_lh[1] * 0.7)
        return speed_lh

    @staticmethod
    def likelihood_rf(detected: bool, strength_dbm: float) -> Tuple[float, float]:
        """RF emission signature."""
        if not detected:
            return (0.30, 0.50)     # Absence of RF = slightly less likely threat
        # Military UAVs use specific frequency bands
        # Commercial drones: 2.4GHz, 5.8GHz
        # Military comms: varies
        if strength_dbm > -60:
            return (0.80, 0.10)     # Strong signal = high threat likelihood
        elif strength_dbm > -90:
            return (0.55, 0.20)
        else:
            return (0.30, 0.25)

    def sequential_bayes_update(
        self, evidence_list: List[Tuple[float, float]], prior: float
    ) -> float:
        """
        Sequential Bayes update (implements Eq. 4 extended to N sources).
        
        For each evidence source (P_threat, P_no_threat):
        posterior = P_threat * prior / (P_threat * prior + P_no_threat * (1 - prior))
        
        The result: combining even weak evidence from multiple domains
        yields a strongly confident final assessment.
        """
        posterior = prior
        for (p_given_threat, p_given_no_threat) in evidence_list:
            # Bayes update
            numerator = p_given_threat * posterior
            denominator = (
                p_given_threat * posterior +
                p_given_no_threat * (1 - posterior)
            )
            if denominator > 0:
                posterior = numerator / denominator
            # Clip to valid probability range
            posterior = np.clip(posterior, 0.001, 0.999)
        return posterior

    def assess_threat(
        self, evidence: ThreatEvidence, track_id: int
    ) -> ThreatAssessment:
        """
        Full Bayesian assessment for a single track.
        Implements the fusion from the spec document:
        
        Vision alone: 60% confidence
        Orbital alone: satellite overhead (passive confirmation)
        COMBINED: crosses 0.95 threshold → RED ALERT
        """

        # Gather likelihood ratios from all domains
        evidence_list = []
        breakdown = {}

        # 1. Vision evidence (Ev in spec document)
        if evidence.vision_confidence > 0:
            lh = self.likelihood_vision(evidence.vision_confidence)
            evidence_list.append(lh)
            breakdown['vision'] = evidence.vision_confidence

        # 2. Orbital evidence (Eo in spec document)
        lh_orb = self.likelihood_orbital(evidence.satellite_elevation_deg)
        evidence_list.append(lh_orb)
        breakdown['orbital_elevation'] = evidence.satellite_elevation_deg

        # 3. Kinematic evidence
        if evidence.track_speed_mps > 0:
            lh_kin = self.likelihood_kinematic(
                evidence.track_speed_mps,
                evidence.kinematic_anomaly
            )
            evidence_list.append(lh_kin)
            breakdown['kinematic_speed'] = evidence.track_speed_mps

        # 4. RF evidence
        if evidence.rf_signature_detected:
            lh_rf = self.likelihood_rf(
                evidence.rf_signature_detected,
                evidence.rf_signal_strength_dbm
            )
            evidence_list.append(lh_rf)
            breakdown['rf_strength'] = evidence.rf_signal_strength_dbm

        # Run sequential Bayes update (implements Eq. 4 extended)
        final_probability = self.sequential_bayes_update(
            evidence_list, self.PRIOR_THREAT
        )

        # Determine threat level
        threat_level = "GREEN"
        for level, (low, high) in self.THRESHOLDS.items():
            if low <= final_probability < high:
                threat_level = level
                break
        if final_probability >= 0.90:
            threat_level = "BLACK"

        # Alert state
        alert_map = {
            "GREEN": "NORMAL",
            "YELLOW": "WATCH",
            "ORANGE": "WARNING",
            "RED": "ALERT",
            "BLACK": "RED_ALERT"   # This triggers the cinematic RED ALERT UI state
        }

        # Recommended action
        action_map = {
            "GREEN": "CONTINUE_MONITORING",
            "YELLOW": "HEIGHTEN_SURVEILLANCE",
            "ORANGE": "REQUEST_IDENTIFICATION",
            "RED": "MOBILIZE_RESPONSE",
            "BLACK": "IMMEDIATE_INTERCEPTION_PROTOCOL"
        }

        return ThreatAssessment(
            threat_id=f"TGT-{track_id:04d}",
            threat_probability=final_probability,
            threat_level=threat_level,
            alert_state=alert_map[threat_level],
            confidence=min(0.99, final_probability + 0.03),
            evidence_breakdown=breakdown,
            recommended_action=action_map[threat_level],
            alert_message=self._generate_alert_message(
                threat_level, evidence, final_probability
            )
        )

    def _generate_alert_message(
        self, level: str, ev: ThreatEvidence, prob: float
    ) -> str:
        if level == "BLACK":
            sat_str = f"SURVEILLANCE PASS ACTIVE — elevation {ev.satellite_elevation_deg:.1f}°. " \
                      if ev.satellite_overhead else ""
            return (
                f"⚡ RED ALERT — THREAT PROBABILITY {prob*100:.1f}%. "
                f"CLASSIFICATION: {ev.detected_class}. "
                f"{sat_str}"
                f"TACTICAL RESPONSE REQUIRED."
            )
        elif level == "RED":
            return f"⚠️ ALERT — {ev.detected_class} detected. P(threat)={prob*100:.1f}%"
        elif level == "ORANGE":
            return f"⚠ WARNING — Unidentified {ev.detected_class}. Monitoring."
        elif level == "YELLOW":
            return f"👁 WATCH — Elevated activity detected."
        return "✓ NOMINAL — No threat detected."
