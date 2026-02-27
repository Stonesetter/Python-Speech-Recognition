"""
=============================================================================
Basic Speech-to-Text with Tkinter Setup Script
=============================================================================
Run this FIRST to install dependencies and download the speech model.

Usage:
    python setup.py              # Install deps + download small model
    python setup.py --large      # Install deps + download large (accurate) model
=============================================================================
"""

import subprocess
import sys
import os
import argparse


def install_packages():
    """Install required Python packages."""
    packages = [
        "vosk",       # Speech recognition engine (offline, fast)
        "pyaudio",    # Microphone audio capture
        "pynput",     # Global keyboard hotkeys
    ]
    
    print("=" * 50)
    print("  Installing Python dependencies...")
    print("=" * 50)
    
    for pkg in packages:
        print(f"\n→ Installing {pkg}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  ⚠ Failed to install {pkg}")
            print(f"  Error: {result.stderr}")
            
            if pkg == "pyaudio":
                print("\n  PyAudio can be tricky on Windows. Try one of these:")
                print("  1. pip install pipwin && pipwin install pyaudio")
                print("  2. Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
                print("  3. conda install pyaudio (if using Anaconda)")
            return False
        else:
            print(f"  ✓ {pkg} installed")
    
    return True


def download_model(model_name):
    """Download and extract a Vosk model."""
    import urllib.request
    import zipfile
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
    
    if os.path.exists(model_path):
        print(f"\n✓ Model already exists: {model_path}")
        return True
    
    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    zip_path = f"{model_name}.zip"
    
    print(f"\n{'=' * 50}")
    print(f"  Downloading: {model_name}")
    print(f"  From: {url}")
    print(f"{'=' * 50}")
    
    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {pct}%  ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=progress)
        print("\n\n  Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(".")
        
        os.remove(zip_path)
        print(f"  ✓ Model ready: {model_path}")
        return True
        
    except Exception as e:
        print(f"\n\n  ✗ Download failed: {e}")
        print(f"  Download manually from: {url}")
        return False


def verify_microphone():
    """Check if a microphone is available."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        
        # Check for input devices
        input_devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                input_devices.append(info["name"])
        
        pa.terminate()
        
        if input_devices:
            print(f"\n✓ Microphone detected:")
            for dev in input_devices[:3]:  # Show first 3
                print(f"  - {dev}")
            if len(input_devices) > 3:
                print(f"  ... and {len(input_devices) - 3} more")
            return True
        else:
            print("\n⚠ No microphone detected!")
            print("  Make sure your mic is plugged in and enabled in Windows Sound settings.")
            return False
            
    except Exception as e:
        print(f"\n⚠ Could not check microphone: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Basic Speech-to-Text with Tkinter Setup")
    parser.add_argument("--large", action="store_true",
                        help="Download the large (more accurate) model instead of small")
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════╗
║          Basic Speech-to-Text with Tkinter — Setup                 ║
║          Speech Recognition Dictation App        ║
╚══════════════════════════════════════════════════╝
    """)
    
    # Step 1: Install packages
    if not install_packages():
        print("\n✗ Setup failed during package installation.")
        print("  Fix the errors above and run setup.py again.")
        sys.exit(1)
    
    # Step 2: Download model
    if args.large:
        model = "vosk-model-en-us-0.22"
        print("\n  Using LARGE model (~1.8GB) — best accuracy")
    else:
        model = "vosk-model-small-en-us-0.15"
        print("\n  Using SMALL model (~50MB) — fast, good accuracy")
    
    if not download_model(model):
        print("\n⚠ Model download failed, but you can download it manually later.")
    
    # Step 3: Check microphone
    verify_microphone()
    
    # Done!
    print(f"""
{'=' * 50}
  ✓ Setup Complete!
{'=' * 50}

  To launch Basic Speech-to-Text with Tkinter:
    python dictator.py --model {model}

  Quick reference:
    Ctrl+Shift+D     Toggle dictation (global hotkey)
    Ctrl+S            Save file
    "period"          Inserts .
    "comma"           Inserts ,
    "new line"        Inserts newline
    "question mark"   Inserts ?

  For the large model (more accurate):
    python setup.py --large
    python dictator.py --model vosk-model-en-us-0.22

  For other languages, download models from:
    https://alphacephei.com/vosk/models
    
  Then run:
    python dictator.py --model <model-directory-name>
{'=' * 50}
""")


if __name__ == "__main__":
    main()
