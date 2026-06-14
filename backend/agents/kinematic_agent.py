"""
KINEMATIC AGENT — Extended Kalman Filter
Translates Equations (2) and (3) from the spec document into working code.

State vector x = [x, y, vx, vy]^T  (Eq. 2)
Prediction:   P_(k|k-1) = F*P_(k-1)*F^T + Q  (Eq. 3)

The magic: When Vision Agent confidence drops to 0 (target behind terrain),
the EKF continues predicting position from physics alone.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
import time


@dataclass
class KinematicState:
    """Full state of a tracked target at one point in time."""
    track_id: int
    timestamp: float

    # EKF State vector: [x, y, vx, vy]
    x: np.ndarray       # Shape (4,) — position + velocity
    P: np.ndarray       # Shape (4,4) — state covariance (uncertainty)

    # Derived outputs
    position: Tuple[float, float] = (0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    heading_deg: float = 0.0

    # Occlusion state
    is_occluded: bool = False
    frames_lost: int = 0
    predicted_reacquisition: Optional[Tuple[float, float]] = None

    # Threat assessment
    threat_label: str = "UNKNOWN"
    confidence: float = 0.0


class ExtendedKalmanFilter:
    """
    EKF for 2D target tracking.
    
    State: x = [x, y, vx, vy]^T  (Eq. 2 from spec)
    
    Constant velocity motion model.
    Non-linear measurement function h(x) = [x, y] (linear in this case,
    but class structure allows extension to non-linear bearing-only tracking).
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,     # 30fps camera assumed
        process_noise: float = 1.0,  # Q matrix scale
        meas_noise: float = 10.0,    # R matrix scale (pixels)
    ):
        self.dt = dt

        # State transition matrix F (Eq. 2 context — how state evolves)
        # x(k) = F * x(k-1) + process_noise
        # [x']   [1 0 dt 0 ] [x ]
        # [y'] = [0 1 0  dt] [y ]
        # [vx']  [0 0 1  0 ] [vx]
        # [vy']  [0 0 0  1 ] [vy]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=float)

        # Observation matrix H — we observe x, y position
        # z = H * x + measurement_noise
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Process noise covariance Q
        # Accounts for unmodeled acceleration
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        self.Q = process_noise * np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ]
        ], dtype=float)

        # Measurement noise covariance R
        self.R = meas_noise * np.eye(2, dtype=float)

        # Identity matrix
        self.I = np.eye(4, dtype=float)

    def init_state(self, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize state from first detection."""
        state = np.array([x, y, 0.0, 0.0], dtype=float)
        # Large initial uncertainty
        cov = np.diag([100.0, 100.0, 1000.0, 1000.0])
        return state, cov

    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        PREDICTION PHASE — Eq. (3) from spec document.
        
        x_pred = F * x
        P_pred = F * P * F^T + Q
        
        Called EVERY frame, even during occlusion.
        This is how we predict WHERE the drone will emerge.
        """
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q     # Eq. (3)
        return x_pred, P_pred

    def update(
        self, x_pred: np.ndarray, P_pred: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        UPDATE PHASE — Called when Vision Agent provides a measurement.
        Standard Kalman update equations.
        
        Innovation:  y = z - H*x_pred
        Kalman Gain: K = P_pred * H^T * (H*P_pred*H^T + R)^-1
        State:       x = x_pred + K*y
        Covariance:  P = (I - K*H) * P_pred
        """
        z = measurement  # [x_pixel, y_pixel]

        # Innovation
        y_innov = z - self.H @ x_pred

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R

        # Kalman Gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Updated state and covariance
        x_upd = x_pred + K @ y_innov
        P_upd = (self.I - K @ self.H) @ P_pred

        return x_upd, P_upd

    def uncertainty_radius(self, P: np.ndarray) -> float:
        """Returns 1-sigma position uncertainty radius in pixels."""
        return float(np.sqrt(P[0, 0] + P[1, 1]))


class MultiTargetTracker:
    """
    Manages multiple concurrent EKF instances (one per tracked target).
    Handles detection → track association, new track initialization,
    and track termination after prolonged occlusion.
    """

    MAX_OCCLUSION_FRAMES = 90   # 3 seconds at 30fps before dropping track
    NEW_TRACK_THRESHOLD = 80.0  # Pixels — max distance to associate detection with track

    def __init__(self, dt: float = 1/30.0):
        self.ekf = ExtendedKalmanFilter(dt=dt)
        self.tracks: dict[int, KinematicState] = {}
        self.next_track_id = 1

    def _distance(self, pos1: tuple, pos2: tuple) -> float:
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def update(self, detections: list) -> List[KinematicState]:
        """
        Full update cycle:
        1. Predict all existing tracks forward
        2. Associate detections to tracks (nearest-neighbor)
        3. Update matched tracks, mark unmatched as occluded
        4. Initialize new tracks for unassociated detections
        5. Drop tracks exceeding max occlusion frames
        """
        # Step 1: Predict all tracks
        for tid, track in self.tracks.items():
            track.x, track.P = self.ekf.predict(track.x, track.P)
            track.is_occluded = True  # Will be set False if matched below

        # Step 2: Associate detections to nearest track
        unmatched_detections = list(detections)
        for tid, track in self.tracks.items():
            predicted_pos = (track.x[0], track.x[1])
            best_det = None
            best_dist = self.NEW_TRACK_THRESHOLD

            for det in unmatched_detections:
                cx, cy = det['bbox_xywh'][0], det['bbox_xywh'][1]
                dist = self._distance(predicted_pos, (cx, cy))
                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            if best_det is not None:
                # Update this track with measurement
                z = np.array([best_det['bbox_xywh'][0], best_det['bbox_xywh'][1]])
                track.x, track.P = self.ekf.update(track.x, track.P, z)
                track.is_occluded = False
                track.frames_lost = 0
                track.confidence = best_det.get('confidence', 0.5)
                track.threat_label = best_det.get('class_name', 'UNKNOWN')
                unmatched_detections.remove(best_det)
            else:
                track.frames_lost += 1

        # Step 3: Initialize new tracks for unmatched detections
        for det in unmatched_detections:
            cx, cy = det['bbox_xywh'][0], det['bbox_xywh'][1]
            x0, P0 = self.ekf.init_state(cx, cy)

            new_track = KinematicState(
                track_id=self.next_track_id,
                timestamp=time.time(),
                x=x0,
                P=P0,
                confidence=det.get('confidence', 0.5),
                threat_label=det.get('class_name', 'UNKNOWN')
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        # Step 4: Drop stale tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if t.frames_lost < self.MAX_OCCLUSION_FRAMES
        }

        # Step 5: Compute derived quantities
        for track in self.tracks.values():
            track.position = (float(track.x[0]), float(track.x[1]))
            track.velocity = (float(track.x[2]), float(track.x[3]))
            track.speed = float(np.sqrt(track.x[2]**2 + track.x[3]**2))
            track.heading_deg = float(np.degrees(np.arctan2(track.x[3], track.x[2])))

            if track.is_occluded:
                # Project predicted reacquisition point
                # Predict 10 frames ahead — this is the "holographic vector line"
                x_future, _ = self.ekf.predict(track.x, track.P)
                for _ in range(10):
                    x_future, _ = self.ekf.predict(x_future, track.P)
                track.predicted_reacquisition = (float(x_future[0]), float(x_future[1]))

        return list(self.tracks.values())
