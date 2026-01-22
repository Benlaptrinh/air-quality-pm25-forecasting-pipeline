"""
Ingest PM2.5 data from OpenAQ API
- Supports multiple sensors
- Fetches hourly data (more rows than daily)
- Saves raw JSON files per sensor
"""
import requests
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found. Check .env file")

HEADERS = {
    "accept": "application/json",
    "X-API-Key": API_KEY
}

# Get sensor IDs (support multiple)
SENSOR_IDS_STR = os.getenv("SENSOR_IDS", "")
SENSOR_IDS = [s.strip() for s in SENSOR_IDS_STR.split(",") if s.strip()]

if not SENSOR_IDS:
    # Fallback to single sensor
    SENSOR_IDS = [int(os.getenv("SENSOR_ID", "5049"))]

LIMIT = int(os.getenv("LIMIT", "100"))

# Date range
DATE_FROM = os.getenv("DATE_FROM", "")  # Format: 2024-01-01
DATE_TO = os.getenv("DATE_TO", "")      # Format: 2025-01-01


def to_utc_iso(dt):
    """Convert datetime to UTC ISO format"""
    if isinstance(dt, str):
        return dt if dt.endswith('Z') else dt + 'Z'
    return dt.replace(microsecond=0).isoformat() + "Z"


def fetch_measurements(sensor_id, interval="hourly", page=1, limit=100):
    """
    Fetch measurements from OpenAQ API

    Args:
        sensor_id: OpenAQ sensor ID
        interval: 'hourly' or 'daily'
        page: Page number
        limit: Records per page (max 100)

    Returns:
        dict: API response
    """
    url = f"{BASE_URL}/sensors/{sensor_id}/measurements"

    params = {
        "limit": limit,
        "page": page,
        "interval": interval  # hourly or daily as query param
    }

    if DATE_FROM:
        params["datetime_from"] = to_utc_iso(DATE_FROM)
    if DATE_TO:
        params["datetime_to"] = to_utc_iso(DATE_TO)

    print(f"   URL: {url}")
    print(f"   Params: {params}")

    response = requests.get(url, headers=HEADERS, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_all_for_sensor(sensor_id, interval="hourly"):
    """
    Fetch all measurements for a single sensor
    """
    all_results = []
    page = 1
    
    print(f"\n📡 Fetching sensor {sensor_id} ({interval})...")
    
    while True:
        try:
            data = fetch_measurements(sensor_id, interval=interval, page=page, limit=LIMIT)
            
            results = data.get("results", [])
            
            if not results:
                break
            
            all_results.extend(results)
            
            meta = data.get("meta", {})
            found = meta.get("found", 0)
            page_limit = meta.get("limit", 100)
            
            print(f"   Page {page}: got {len(results)} records (total: {found})")
            
            # Stop if we've got all pages
            if len(results) < page_limit:
                break
            
            page += 1
            time.sleep(0.3)  # Rate limiting
            
        except Exception as e:
            print(f"   ❌ Error on page {page}: {e}")
            break
    
    return all_results


def save_sensor_data(data, sensor_id, interval):
    """
    Save sensor data to JSON file
    """
    os.makedirs("data/raw", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"data/raw/sensor_{sensor_id}_{interval}_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "sensor_id": sensor_id,
            "interval": interval,
            "date_from": DATE_FROM,
            "date_to": DATE_TO,
            "count": len(data),
            "results": data
        }, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Saved {len(data)} records to {filename}")
    return filename


def main():
    """
    Main function to ingest data from multiple sensors
    """
    print("=" * 60)
    print("🚀 OpenAQ Multi-Sensor Ingest")
    print("=" * 60)
    print(f"Sensors: {SENSOR_IDS}")
    print(f"Date range: {DATE_FROM or 'beginning'} to {DATE_TO or 'now'}")
    print(f"Interval: hourly (to maximize rows)")
    print("=" * 60)
    
    total_records = 0
    files_created = []
    
    for sensor_id in SENSOR_IDS:
        # Fetch hourly data (more rows than daily!)
        results = fetch_all_for_sensor(sensor_id, interval="hourly")
        
        if results:
            filename = save_sensor_data(results, sensor_id, "hourly")
            files_created.append(filename)
            total_records += len(results)
        else:
            print(f"   ⚠️ No data for sensor {sensor_id}")
    
    print("\n" + "=" * 60)
    print(f"🎉 INGEST COMPLETE!")
    print(f"   Total sensors: {len(SENSOR_IDS)}")
    print(f"   Total records: {total_records}")
    print(f"   Files created: {len(files_created)}")
    print("=" * 60)
    
    return files_created


if __name__ == "__main__":
    main()
