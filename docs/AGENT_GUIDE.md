# Agent Guide

The Sudarshan system is driven by four independent agents.

## 1. Vision Agent (YOLOv10)
- **Role:** Extracts bounding boxes from pixel arrays.
- **Hardware:** Optimized for GPU execution (CUDA/TensorRT) but gracefully falls back to CPU.
- **Output:** `DetectionBase` (confidence, coordinates, class label).

## 2. Kinematic Agent (EKF)
- **Role:** Maintains target continuity.
- **Logic:** Matches new YOLO detections to existing tracks using Mahalanobis distance and Intersection over Union (IoU) with Non-Maximum Suppression (NMS).
- **Occlusion Handling:** If a target vanishes, the EKF enters "Coast Mode," predicting the trajectory up to `MAX_OCCLUSION_FRAMES` before dropping the track.

## 3. Orbital Agent (SGP4)
- **Role:** Space Domain Awareness.
- **Logic:** Loads the offline TLE cache and continuously calculates the elevation of known ISR (Intelligence, Surveillance, Reconnaissance) satellites over the observer's GPS coordinates.
- **Threat Trigger:** Any satellite above `SURVEILLANCE_ELEVATION_THRESHOLD` (e.g., 15°) with an active line of sight is flagged.

## 4. Tactical Fusion Agent
- **Role:** The Brain.
- **Logic:** Consumes the outputs of the previous three agents. If a drone is detected moving fast (Kinematic Risk) while a satellite is overhead (Orbital Risk), the Fusion engine spikes the final Threat Probability output via Bayesian inference.
