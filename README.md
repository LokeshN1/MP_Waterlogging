# 🌊 Waterlogging Detection System

An AI-powered real-time waterlogging detection system that uses your phone as a live IP camera, captures frames periodically, classifies them using a TensorFlow/Keras MobileNetV1 model, and displays results on a live browser-based dashboard.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Step 1 — Clone the Repository](#step-1--clone-the-repository)
- [Step 2 — Set Up Python Environment](#step-2--set-up-python-environment)
- [Step 3 — Download the Model](#step-3--download-the-model)
- [Step 4 — Connect Your Phone Camera](#step-4--connect-your-phone-camera)
- [Step 5 — Run the System](#step-5--run-the-system)
- [Scripts Reference](#scripts-reference)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- 📷 Live frame capture from Android or iOS phone camera over Wi-Fi
- 🤖 AI classification — **Waterlogged** vs **Non-Waterlogged** with confidence score
- 🖥️ Browser dashboard at `http://localhost:5000` with live preview, history & alerts
- 🔔 Configurable alert threshold and cooldown
- 📂 Analyse any image manually via the dashboard upload button

---

## 📁 Project Structure

```
MP_Waterlogging/
├── model/
│   ├── waterlogging_mobilenet_v1.h5     ← main model (tracked in git)
│   └── model.weights.h5                 ← NOT in git (download via script)
├── analyis/                             ← sample analysis images
├── snippets/                            ← auto-created; captured frames go here
├── snippet_capture.py                   ← captures frames from phone camera
├── snippets_detector_gui.py             ← browser dashboard (Flask)
├── detect_snippets.py                   ← CLI-only frame monitor
├── waterlogging_detection.py            ← model loading & prediction logic
├── test_model.py                        ← quick test against sample images
├── download_model.py                    ← downloads model weights from Drive
├── requirements.txt
└── README.md
```

---

## ✅ Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.10 or 3.11 recommended |
| **Phone** | Android or iOS, on the **same Wi-Fi** as your PC |
| **Camera app** | Android: [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) · iOS: [IP Camera Lite](https://apps.apple.com/app/ip-camera-lite/id1343055863) |
| **Internet** | Required once to download model weights |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Kunalwaldia8/MP_Waterlogging.git
cd MP_Waterlogging
```

---

## Step 2 — Set Up Python Environment

Create and activate a virtual environment, then install all dependencies.

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## Step 3 — Download the Model

The model weights file (~195 MB) is too large for GitHub and is **not included** in the repo. Run this script once to download it automatically from Google Drive:

```bash
python download_model.py
```

You should see:
```
⬇️  Downloading model weights via gdown...
✅  Downloaded successfully → model/model.weights.h5 (194.5 MB)
```

> If the script fails, download manually from [this Google Drive link](https://drive.google.com/file/d/1xEwaTW40q26I2tFFIQ4SC5M74zRR2liK/view?usp=sharing) and place the file at `model/model.weights.h5`.

---

## Step 4 — Connect Your Phone Camera

### Android — IP Webcam

1. Install **[IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)** from the Play Store
2. Open the app → scroll to the bottom → tap **"Start server"**
3. Note the IP shown (e.g. `192.168.1.5:8080`)
4. Open `snippet_capture.py` and update line 6:
   ```python
   CAMERA_URL_BASE = 'http://192.168.1.5:8080/shot.jpg'
   ```

### iOS — IP Camera Lite

1. Install **[IP Camera Lite](https://apps.apple.com/app/ip-camera-lite/id1343055863)** from the App Store
2. Open the app → tap **"Start"**
3. Note the IP shown (e.g. `192.168.1.7:8080`)
4. Open `snippet_capture.py` and update line 6:
   ```python
   CAMERA_URL_BASE = 'http://192.168.1.7:8080/photo.jpg'
   ```
   *(Note: iOS uses `/photo.jpg`, Android uses `/shot.jpg`)*

### Verify connection

Open the camera URL in your PC browser — you should see a live photo from your phone. If it shows a username/password prompt, disable authentication in the app settings.

---

## Step 5 — Run the System

You need **two terminals** running simultaneously.

### Terminal 1 — Start capturing frames

```bash
# Make sure your venv is active
source venv/bin/activate        # Linux/macOS
# or: venv\Scripts\activate     # Windows

python snippet_capture.py
```

Expected output:
```
Capturing from http://192.168.1.5:8080/shot.jpg every 3 seconds...
Saved: snippets/frame_20260509_001500.jpg
Saved: snippets/frame_20260509_001503.jpg
...
```

### Terminal 2 — Start the dashboard

```bash
source venv/bin/activate
python snippets_detector_gui.py
```

Expected output:
```
🌊 Flood Vision Console running at  http://127.0.0.1:5000
   Press Ctrl+C to stop.
```

Your browser will open automatically at **http://127.0.0.1:5000** showing:
- 📸 Live preview of the latest captured frame
- 🤖 AI verdict (Waterlogged / Non-Waterlogged) with confidence %
- 📊 Prediction history table
- 🔔 Alert log with configurable threshold

### Stop the system

Press `Ctrl+C` in both terminals.

---

## 🧪 Quick Test (No Phone Needed)

To verify the model works without a phone, run:

```bash
python test_model.py
```

This classifies the sample images in the `analyis/` folder and prints results:
```
img1.jpg  →  Non-Waterlogged   Confidence: 94.07%
img3.jpg  →  Waterlogged       Confidence: 95.84%
```

---

## 📜 Scripts Reference

| Script | Purpose |
|---|---|
| `snippet_capture.py` | Fetches a frame from the phone camera every 3 seconds and saves to `snippets/` |
| `snippets_detector_gui.py` | Runs the Flask web dashboard at port 5000 |
| `detect_snippets.py` | Terminal-only version — watches `snippets/` and prints predictions |
| `waterlogging_detection.py` | Core module — model loading and `predict_waterlogging()` function |
| `test_model.py` | Runs inference on the sample images in `analyis/` |
| `download_model.py` | Downloads `model/model.weights.h5` from Google Drive |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure your virtual environment is activated and `pip install -r requirements.txt` was run |
| Camera URL not reachable | Ensure phone and PC are on **the same Wi-Fi network** |
| Browser shows login prompt | Disable username/password in the camera app settings |
| `model.weights.h5` not found | Run `python download_model.py` |
| Dashboard shows "No frames found" | Start `snippet_capture.py` first and wait a few seconds |
| TensorFlow GPU warnings | These are harmless — the model runs fine on CPU |
| Push rejected (file too large) | `model/` files are in `.gitignore` — do not add them to git |

---

## 📄 License

No license file is currently included. Add one (e.g. MIT) before public distribution.
