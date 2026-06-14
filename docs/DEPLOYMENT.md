# Deployment & Operations Guide

## Air-Gapped Philosophy
Project Sudarshan is designed for highly secure edge nodes where internet access is either physically severed or heavily compromised by Electronic Warfare (EW).

### 1. Preparing the Cache (Online)
Before deploying to the edge, run these commands while connected to the internet to pre-fetch all models and orbital data:
```bash
# Downloads yolov10s.pt
python scripts/download_models.py

# Downloads Active Satellite TLEs to backend/data/tle_cache.txt
python scripts/fetch_tle.py
```

### 2. Physical Transport
Move the entire `project-sudarshan` directory onto a secure USB drive and transfer it to the edge hardware.

### 3. Execution (Offline)
Run the master boot script. The system will detect the local caches and boot without attempting any external network calls.
```bash
./scripts/demo_start.sh
```

### 4. Headless Mode
If the edge node has no display (e.g., a rackmount server at a radar site), the React frontend can be accessed by any tablet or computer on the local intranet by navigating to the server's IP address on port 5173.
