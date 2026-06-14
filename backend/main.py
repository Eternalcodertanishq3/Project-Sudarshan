"""
PROCESS C — FastAPI WebSocket Event Bus
Broadcasts fused threat intelligence to React frontend at 60Hz.
"""

import asyncio
import multiprocessing as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from orchestrator.event_bus import broadcaster

app = FastAPI(title="Project Sudarshan C4ISR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set dynamically by process_manager
broadcast_queue: mp.Queue = None

@app.websocket("/ws/tactical-feed")
async def tactical_feed(websocket: WebSocket):
    await broadcaster.register(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.deregister(websocket)

@app.post("/api/scenario/{scenario_name}")
async def inject_scenario(scenario_name: str):
    from simulation.demo_scenarios import SCENARIOS
    if scenario_name in SCENARIOS:
        if broadcast_queue:
            broadcast_queue.put(SCENARIOS[scenario_name])
        return {"status": "injected", "scenario": scenario_name}
    return {"error": f"Unknown scenario: {scenario_name}"}

@app.get("/api/status")
async def system_status():
    return {
        "system": "PROJECT_SUDARSHAN",
        "status": "OPERATIONAL",
        "domains": ["AIR", "LAND", "SEA", "SPACE"],
        "opsec": "AIR_GAPPED",
        "connected_clients": len(broadcaster.clients)
    }

async def broadcast_loop(queue: mp.Queue):
    """Core broadcast loop at 60Hz."""
    while True:
        try:
            if not queue.empty():
                payload = queue.get_nowait()
                await broadcaster.broadcast(payload)
        except Exception:
            pass
        await asyncio.sleep(1.0 / 60.0)

