#!/bin/bash
# PROJECT SUDARSHAN — Demo Launch Script
# Run: chmod +x scripts/demo_start.sh && ./scripts/demo_start.sh

set -e

echo "╔══════════════════════════════════════════╗"
echo "║     PROJECT SUDARSHAN — INITIALIZING     ║"
echo "║     Air · Land · Sea · Space             ║"
echo "╚══════════════════════════════════════════╝"

# Check TLE cache
if [ ! -f "backend/data/tle_cache.txt" ]; then
  echo "[*] Pre-fetching TLE catalog for air-gapped operation..."
  python scripts/fetch_tle.py
fi

# Check YOLO weights
if [ ! -f "yolov10s.pt" ]; then
  echo "[*] Downloading YOLOv10 weights..."
  python scripts/download_models.py
fi

# Launch backend (all 3 processes)
echo "[*] Starting Process A (Vision), B (Intel), C (Event Bus)..."
cd backend
# Running process_manager to start all agents
python -m orchestrator.process_manager &
BACKEND_PID=$!
cd ..

# Launch frontend
echo "[*] Starting React dashboard..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

sleep 3

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     SUDARSHAN ONLINE                     ║"
echo "║     Dashboard: http://localhost:5173      ║"
echo "║     API:       http://localhost:8000      ║"
echo "║     WebSocket: ws://localhost:8000/ws/   ║"
echo "╚══════════════════════════════════════════╝"

# Open browser
xdg-open http://localhost:5173 2>/dev/null || open http://localhost:5173 2>/dev/null || true

# Wait and cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Sudarshan offline.'" EXIT
wait
