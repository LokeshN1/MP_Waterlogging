# Waterlogging Detection System

Real-time waterlogging detection pipeline using:
- An Android or iOS phone as an IP camera,
- Periodic frame capture to disk,
- TensorFlow/Keras (MobileNetV1) classification,
- A browser-based Flask dashboard for live monitoring.

## Features

- Live frame capture from phone camera (`/shot.jpg` endpoint).
- GUI dashboard with prediction history and alert controls.
- CLI detector for terminal-only monitoring.
- Model-loading compatibility handling for different Keras versions.

## Project Layout

```
Major Project/
|-- model/
|   |-- waterlogging_inceptionv3.keras
|   `-- waterlogging_mobilenet_v1.h5
|-- analyis/
|-- resultImg2/
|-- snippet_capture.py
|-- snippets_detector_gui.py
|-- detect_snippets.py
|-- waterlogging_detection.py
|-- test_model.py
|-- requirements.txt
|-- .gitignore
|-- .gitattributes
`-- README.md
```

## Prerequisites

- Python 3.10 or 3.11 recommended
- **Android**: [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app
- **iOS**: [IP Camera Lite](https://apps.apple.com/app/ip-camera-lite/id1343055863) app
- Phone and computer connected to the same Wi-Fi network

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the model weights:

```bash
python download_model.py
```

## ⬇️ Download Model Weights (Required)

The model file (`model/model.weights.h5`, ~195 MB) is too large for GitHub and is **not included in the repo**.

**After cloning, run this once:**

```bash
python download_model.py
```

This will automatically download the weights from Google Drive.

> If the script fails, download manually from the link in `download_model.py` and place the file at `model/model.weights.h5`.

## Configure Camera URL

In `snippet_capture.py`, update:

```python
CAMERA_URL_BASE = "http://<phone-ip>:8080/shot.jpg"
```

Example:

```python
CAMERA_URL_BASE = "http://192.168.1.6:8080/shot.jpg"
```

## How to Run This Project

Follow these steps in order.

1. Install IP Webcam from play store.then Start the IP Webcam server on your Android phone.(jab tum ip webcam app ko install kr doge to use scroll krna last m option hoga start server use start krna tumhara camera open ho jayega. fir wha bottom me ip address likha hoga tumhe uss ip address ko note krna h)
2. Make sure phone and laptop are on the same Wi-Fi.
3. Confirm `CAMERA_URL_BASE` in `snippet_capture.py` points to your phone IP.(`snippet_capture.py` file me jha per `CAMERA_URL_BASE` likha hoga wha per wo ip address dal dena)

### Step 1: Open project in terminal

```bash
cd "c:\Users\lokes\Desktop\Major Project"
```

### Step 2: Create and activate virtual environment (recommended)

PowerShell:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Command Prompt (CMD):

```bash
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Step 3: Run capture and detection (two terminals)

Terminal 1 (frame capture):
iss files s live photo capture hogi, wha per ek `snippets` folder ban jayega 
```bash
python snippet_capture.py
```

Terminal 2 (Web dashboard — opens in browser automatically):
isme tum dekh paoge result at http://127.0.0.1:5000
```bash
python snippets_detector_gui.py
```



### How to stop

- Press `Ctrl+C` in both terminal windows.

## Scripts

- `snippet_capture.py`: captures frames every few seconds into `snippets/`.
- `snippets_detector_gui.py`: live dashboard, alerting, and history.
- `detect_snippets.py`: watches `snippets/` and prints predictions.
- `waterlogging_detection.py`: model loading and `predict_waterlogging()`.
- `test_model.py`: quick test against images inside `analyis/`.

## Troubleshooting

- Cannot connect to camera URL:
   Ensure phone and PC are on same Wi-Fi and IP is correct.
- GUI shows no frames:
   Start `snippet_capture.py` first and verify `snippets/` is being populated.
- TensorFlow model load errors:
   Use Python 3.10/3.11 and install dependencies from `requirements.txt`.
- Push rejected by GitHub due to file size:
   Use Git LFS for files in `model/`.

## License

No license file is currently included. Add one (for example, MIT) before public distribution if needed.
