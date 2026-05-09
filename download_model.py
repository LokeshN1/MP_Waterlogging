"""
download_model.py
-----------------
Downloads the waterlogging model weights from Google Drive if not already present.

Usage:
    python download_model.py

After cloning the repo, run this once before running any detection scripts.
"""

import os
import sys

# ── Configuration ──────────────────────────────────────────────────────────────
# Paste your Google Drive shareable link here after uploading model.weights.h5
# It should look like: https://drive.google.com/file/d/XXXX/view?usp=sharing
GDRIVE_SHARE_URL = "https://drive.google.com/file/d/1xEwaTW40q26I2tFFIQ4SC5M74zRR2liK/view?usp=sharing"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.weights.h5")
# ──────────────────────────────────────────────────────────────────────────────


def _extract_file_id(share_url: str) -> str:
    """Extract the file ID from a Google Drive shareable URL."""
    # Handles both /file/d/<id>/view and /open?id=<id> formats
    if "/file/d/" in share_url:
        return share_url.split("/file/d/")[1].split("/")[0]
    if "id=" in share_url:
        return share_url.split("id=")[1].split("&")[0]
    raise ValueError(
        f"Could not extract file ID from URL: {share_url}\n"
        "Make sure it's a valid Google Drive shareable link."
    )


def download_model() -> None:
    if GDRIVE_SHARE_URL == "PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE":
        print(
            "❌  Google Drive URL not configured.\n"
            "    Open download_model.py and set GDRIVE_SHARE_URL to your shareable link."
        )
        sys.exit(1)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)
        print(f"✅  Model already exists at {MODEL_PATH} ({size_mb:.1f} MB). Nothing to do.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Try gdown first (best for large Drive files), fall back to requests
    try:
        import gdown  # type: ignore
        file_id = _extract_file_id(GDRIVE_SHARE_URL)
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"⬇️   Downloading model weights via gdown...")
        gdown.download(download_url, MODEL_PATH, quiet=False)
    except ImportError:
        print("ℹ️   gdown not found, falling back to requests (may fail for large files).")
        print("    Install gdown for reliable downloads:  pip install gdown")
        _download_with_requests()

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)
        print(f"\n✅  Downloaded successfully → {MODEL_PATH} ({size_mb:.1f} MB)")
    else:
        print("\n❌  Download failed. Please download manually and place at:")
        print(f"    {MODEL_PATH}")
        sys.exit(1)


def _download_with_requests() -> None:
    import requests

    file_id = _extract_file_id(GDRIVE_SHARE_URL)
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = session.get(url, stream=True)
    # Handle Google's large-file warning page
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
        response = session.get(url, stream=True)

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    Progress: {pct:.1f}% ({downloaded/(1024**2):.1f}/{total/(1024**2):.1f} MB)", end="")
    print()


if __name__ == "__main__":
    download_model()
