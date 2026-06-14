"""
ORCHESTRATOR — Project Sudarshan Main Entry Point
Ties together Vision, Kinematic, Orbital, and Tactical agents across multiple processes.
"""

import multiprocessing as mp
import time
import asyncio
import uvicorn
import cv2
import numpy as np

from agents.vision_agent import VisionAgent
from agents.kinematic_agent import MultiTargetTracker
from agents.orbital_agent import OrbitalAgent
from agents.tactical_agent import BayesianFusionEngine, ThreatEvidence
import main as app_main
from orchestrator.shared_state import shared_state
from utils.tactical_logger import tactical_log


def process_b_worker(detection_queue, broadcast_queue, stop_event):
    """
    PROCESS B: Kinematic Tracking, Orbital Math, and Tactical Fusion.
    Runs on CPU. Consumes detections from Process A, pushes JSON to Process C.
    """
    tactical_log.info("Process B (Tracking & Fusion) started.")
    
    # Initialize Agents
    tracker = MultiTargetTracker(dt=1/30.0)
    orbital = OrbitalAgent()
    tactical = BayesianFusionEngine()
    
    # Load TLE data synchronously for the worker
    asyncio.run(orbital.load_tle_catalog())
    
    while not stop_event.is_set():
        try:
            # 1. Get detections from Vision Agent (Process A)
            try:
                vision_data = detection_queue.get(timeout=0.05)
                detections = vision_data.get('detections', [])
            except Exception:
                detections = []
            
            # 2. Update Kinematic Tracks
            active_tracks = tracker.update(detections)
            
            # 3. Update Orbital Passes
            overhead_passes = orbital.scan_overhead()
            
            # 4. Tactical Fusion
            fused_threats = []
            highest_threat_level = 0.0
            
            for track in active_tracks:
                # Compile evidence for this track
                evidence = ThreatEvidence(
                    vision_confidence=track.confidence if not track.is_occluded else 0.0,
                    detected_class=track.threat_label,
                    track_speed_mps=track.speed,
                    track_heading_deg=track.heading_deg,
                    is_occluded=track.is_occluded
                )
                
                # If there's an active ISR pass, add orbital evidence
                isr_passes = [p for p in overhead_passes if p.is_threat]
                if isr_passes:
                    best_pass = isr_passes[0]
                    evidence.satellite_overhead = True
                    evidence.satellite_elevation_deg = best_pass.elevation_deg
                    evidence.surveillance_risk = best_pass.surveillance_risk
                
                # Run Bayesian Fusion
                assessment = tactical.assess_threat(evidence, track.track_id)
                highest_threat_level = max(highest_threat_level, assessment.threat_probability)
                
                fused_threats.append({
                    "track_id": track.track_id,
                    "position": track.position,
                    "velocity": track.velocity,
                    "speed": track.speed,
                    "heading": track.heading_deg,
                    "is_occluded": track.is_occluded,
                    "predicted_reacquisition": track.predicted_reacquisition,
                    "assessment": {
                        "threat_probability": assessment.threat_probability,
                        "threat_level": assessment.threat_level,
                        "alert_state": assessment.alert_state,
                        "alert_message": assessment.alert_message,
                        "recommended_action": assessment.recommended_action
                    }
                })
            
            # 5. Build Final Payload and Push to Broadcast Queue
            payload = {
                "timestamp": time.time(),
                "global_threat_level": highest_threat_level,
                "tracks": fused_threats,
                "satellites": [
                    {
                        "name": p.name,
                        "elevation": p.elevation_deg,
                        "azimuth": p.azimuth_deg,
                        "risk": p.surveillance_risk,
                        "is_threat": p.is_threat
                    } for p in overhead_passes
                ]
            }
            
            try:
                # Remove old payload if queue is full to maintain low latency
                if broadcast_queue.full():
                    broadcast_queue.get_nowait()
                broadcast_queue.put_nowait(payload)
            except Exception:
                pass
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            tactical_log.error(f"Process B Error: {e}")
            time.sleep(0.1)

def run_fastapi(broadcast_queue):
    """PROCESS C: FastAPI Event Bus"""
    tactical_log.info("Process C (Event Bus) starting on port 8000...")
    app_main.broadcast_queue = broadcast_queue
    
    # Start the broadcast loop in asyncio
    @app_main.app.on_event("startup")
    async def startup_event():
        asyncio.create_task(app_main.broadcast_loop(broadcast_queue))
        
    uvicorn.run(app_main.app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    print("=======================================")
    print(" PROJECT SUDARSHAN C4ISR INITIALIZING  ")
    print("=======================================")
    
    mp.set_start_method('spawn', force=True)
    
    # 1. Start Process C (Event Bus)
    process_c = mp.Process(target=run_fastapi, args=(shared_state.broadcast_queue,))
    process_c.start()
    
    # 2. Start Process B (Kinematic/Orbital/Tactical)
    process_b = mp.Process(target=process_b_worker, args=(shared_state.detection_queue, shared_state.broadcast_queue, shared_state.stop_event))
    process_b.start()
    
    # 3. Start Process A (Vision Engine)
    vision_agent = VisionAgent(use_gpu=False)
    process_a = mp.Process(target=vision_agent.run, args=(shared_state.frame_queue, shared_state.detection_queue, shared_state.stop_event))
    process_a.start()
    
    tactical_log.info("All processes started. Capturing camera feed...")
    
    # Camera capture loop (Main Process)
    cap = cv2.VideoCapture(0)
    frame_id = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue
                
            frame_id += 1
            
            try:
                if shared_state.frame_queue.full():
                    shared_state.frame_queue.get_nowait()
                shared_state.frame_queue.put_nowait({
                    'frame': frame,
                    'frame_id': frame_id,
                    'timestamp': time.time()
                })
            except Exception:
                pass
                
            time.sleep(1/30.0)
            
    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutting down Project Sudarshan...")
        shared_state.stop_event.set()
        cap.release()
        process_a.terminate()
        process_b.terminate()
        process_c.terminate()
        print("[Orchestrator] Shutdown complete.")
