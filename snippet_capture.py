import requests
import time
import os
import random

CAMERA_URL_BASE = 'http://192.168.1.6:8080/shot.jpg'  # change to your camera's URL

CAPTURE_INTERVAL_SEC = 3
OUTPUT_DIR = 'snippets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Capturing from {CAMERA_URL_BASE} every {CAPTURE_INTERVAL_SEC} seconds...")

try:
    while True:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f"frame_{timestamp}.jpg")
        # Add a random query param to bust cache
        url = f"{CAMERA_URL_BASE}?t={random.randint(0,999999)}"
        try:
            img_data = requests.get(url, timeout=5).content
            with open(filename, 'wb') as handler:
                handler.write(img_data)
            print(f"Saved: {filename}")
        except Exception as e:
            print("[Warning] Could not fetch image:", e)
        time.sleep(CAPTURE_INTERVAL_SEC)
except KeyboardInterrupt:
    print("\nStopped by user.")
print("All done!")
