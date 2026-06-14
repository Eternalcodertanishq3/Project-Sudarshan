import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import List
from utils.tactical_logger import tactical_log

class SudarshanBroadcaster:
    """Manages WebSocket clients and broadcasts the unified tactical picture."""

    def __init__(self):
        self.clients: List[WebSocket] = []

    async def register(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)
        tactical_log.info(f"Client connected. Total: {len(self.clients)}")

    def deregister(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)
        tactical_log.info(f"Client disconnected. Total: {len(self.clients)}")

    async def broadcast(self, payload: dict):
        if not self.clients:
            return
        message = json.dumps(payload)
        dead_clients = []
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                dead_clients.append(client)
        for dc in dead_clients:
            self.deregister(dc)

broadcaster = SudarshanBroadcaster()
