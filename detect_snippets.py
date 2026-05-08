import os
import time

from waterlogging_detection import ModelUnavailableError, predict_waterlogging

# Folder containing snippets
SNIPPETS_DIR = 'snippets'
os.makedirs(SNIPPETS_DIR, exist_ok=True)

def predict_snippet(img_path):
    label, confidence, _ = predict_waterlogging(img_path)
    print(f"Detected: {label} (Confidence: {round(confidence * 100, 2)}%)")
    return label, confidence

# Monitor the folder and classify each image
processed = set()
print("Monitoring snippets folder for new images...")

while True:
    files = sorted(os.listdir(SNIPPETS_DIR))
    for fname in files:
        path = os.path.join(SNIPPETS_DIR, fname)
        if path.endswith('.jpg') and path not in processed:
            print(f"\nProcessing: {fname}")
            try:
                predict_snippet(path)
            except ModelUnavailableError as e:
                print(f"[Model Error] {e}")
            except Exception as e:
                print(f"[Detection Error] {e}")
            processed.add(path)
    # Check for new images every 5 seconds
    time.sleep(5)
