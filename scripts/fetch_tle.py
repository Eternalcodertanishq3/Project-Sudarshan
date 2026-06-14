import asyncio
import httpx
from pathlib import Path
import os

async def cache_tle():
    """Fetches TLE cache from Celestrak for Air-Gapped Demo operations."""
    url = "https://celestrak.org/SOCRATES/gp.php?GROUP=active&FORMAT=tle"
    cache_path = Path("backend/data/tle_cache.txt")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching TLE catalog from {url}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
            with open(cache_path, "w") as f:
                f.write(response.text)
                
        print(f"Successfully cached TLE data to {cache_path}")
    except Exception as e:
        print(f"Failed to fetch TLE data: {e}")
        
if __name__ == "__main__":
    asyncio.run(cache_tle())
