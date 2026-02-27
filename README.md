# Basic Speech-to-Text with Tkinter

A Windows desktop dictation app built in Python, similar to Windows' built-in Win+H feature but with offline processing, voice commands, and multi-language support.

## Quick Start

```bash
# 1. Install dependencies and download the English model
python setup.py

# 2. Launch the app
python dictator.py --model vosk-model-small-en-us-0.15
\*note: make sure that you include the entire filepath where you downloaded the vosk folder at the top of the window, e.g. C:\Windows\User\Documents\ProjectName\
```

That's it! Click **🎤 Start Dictation** or press **Ctrl+Shift+D** and start talking.

## Requirements

- **Python 3.8+** (3.10+ recommended)
- **Windows 10/11** (also works on WSL with PulseAudio, and Linux)
- A microphone

## Features

- **Real-time dictation** — see words appear as you speak
- **Offline processing** — no internet required, your voice data stays private
- **Voice commands** — say "period", "comma", "new line", etc.
- **Global hotkey** — Ctrl+Shift+D toggles dictation even when minimized
- **Full text editor** — open, edit, save text files
- **Dark theme** — easy on the eyes
- **Multi-language ready** — swap models to dictate in other languages

## Voice Commands

| Say This | You Get |
|---|---|
| "period" or "full stop" | . |
| "comma" | , |
| "question mark" | ? |
| "exclamation mark" | ! |
| "colon" | : |
| "semicolon" | ; |
| "new line" | (line break) |
| "new paragraph" | (double line break) |
| "open parenthesis" | ( |
| "close parenthesis" | ) |
| "open quote" | " |
| "close quote" | " |
| "dash" | — |
| "hyphen" | - |

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+Shift+D | Toggle dictation (global) |
| Ctrl+S | Save file |
| Ctrl+O | Open file |
| Ctrl+N | New file |

## Models

### English Models (download from [Vosk Models](https://alphacephei.com/vosk/models))

| Model | Size | Accuracy | Speed |
|---|---|---|---|
| vosk-model-small-en-us-0.15 | ~50MB | Good | Fast |
| vosk-model-en-us-0.22 | ~1.8GB | Great | Medium |
| vosk-model-en-us-0.42-gigaspeech | ~2.3GB | Best | Slower |

### Other Languages

Download any Vosk model and point the app at it:

```bash
python dictator.py --model vosk-model-small-es-0.42    # Spanish
python dictator.py --model vosk-model-small-fr-0.22    # French
python dictator.py --model vosk-model-small-de-0.15    # German
python dictator.py --model vosk-model-small-ja-0.22    # Japanese
python dictator.py --model vosk-model-small-zh-cn-0.22 # Chinese
```

Full list: https://alphacephei.com/vosk/models

---

## 🎓 How It Works — Learning Guide

### The Big Picture

Speech recognition converts sound waves into text. Here's the pipeline:

```
Microphone → Audio Bytes → Feature Extraction → Neural Network → Language Model → Text
```

### 1. Audio Capture (PyAudio)

Your microphone converts air pressure changes into electrical signals, which get digitized into numbers. We configure PyAudio to capture:

- **16000 Hz sample rate** — we take 16,000 measurements per second. Human speech is mostly in the 300-3400 Hz range, and by the Nyquist theorem, we need at least 2x the highest frequency to capture it accurately. 16kHz is the standard for speech recognition.

- **16-bit samples** — each measurement is stored as a 16-bit integer (-32768 to 32767), giving us 65,536 possible amplitude values. This is CD-quality resolution.

- **Mono channel** — speech recognition doesn't benefit from stereo, so we save bandwidth with a single channel.

- **4096 frame chunks** — we read 4096 samples at a time (≈0.25 seconds of audio). This balances latency (how quickly we respond) vs stability (fewer buffer overflows).

### 2. Speech Recognition (Vosk/Kaldi)

Vosk is built on the Kaldi speech recognition toolkit. Here's what happens inside:

**a) Feature Extraction (MFCC)**
Raw audio → Mel-Frequency Cepstral Coefficients. This transforms the audio into a compact representation that captures the spectral characteristics of speech, similar to how your ear processes sound.

**b) Acoustic Model**
A deep neural network that maps audio features → phonemes (the basic units of speech). For example, the word "cat" = /k/ + /æ/ + /t/.

**c) Language Model**
A statistical model that predicts which words are likely to follow each other. This is what makes "recognize speech" more likely than "wreck a nice beach" even though they sound similar.

**d) Decoder**
Combines the acoustic model and language model using a search algorithm (beam search) to find the most likely sequence of words.

### 3. Threading Architecture

The app uses 3 threads:

```
Thread 1: GUI (Tkinter event loop)
  └── Handles button clicks, text editing, display updates
  └── Polls queues every 50ms for new text

Thread 2: Audio Capture (AudioEngine)
  └── Reads microphone → feeds to Vosk → puts results in queues
  └── Runs continuously while dictation is active

Thread 3: Global Hotkey (pynput)
  └── Monitors keyboard for Ctrl+Shift+D
  └── Triggers dictation toggle on GUI thread via after()
```

**Why threading?** The GUI event loop and audio capture both need to run continuously. If we did audio capture on the GUI thread, the window would freeze while listening. Threading lets them run simultaneously.

**Why queues?** Tkinter is NOT thread-safe — you can't update widgets from other threads. The Queue class is thread-safe, so the audio thread puts text into the queue, and the GUI thread reads from it safely.

### 4. Text Processing

The TextProcessor class handles the gap between raw speech and formatted text:

- **Punctuation commands**: "period" → "."
- **Formatting commands**: "new line" → "\n"
- **Auto-capitalization**: After periods, question marks, and new lines
- **Smart spacing**: No space before punctuation, space between words

### 5. How to Extend This

**Add new voice commands:**
Edit the `PUNCTUATION` and `COMMANDS` dictionaries in the `TextProcessor` class.

**Add new languages:**
1. Download a Vosk model for your target language
2. Run: `python dictator.py --model vosk-model-small-XX-0.XX`
3. Add language-specific punctuation commands to TextProcessor

**Add speaker identification:**
Vosk supports speaker diarization (telling voices apart). Look into `vosk.SpkModel`.

**Add custom vocabulary:**
You can create a grammar-restricted recognizer that only listens for specific words — useful for command interfaces.

---

## Troubleshooting

### "No module named pyaudio"
```bash
# Try these in order:
pip install pyaudio
# If that fails:
pip install pipwin && pipwin install pyaudio
# If still failing, install Microsoft C++ Build Tools first
```

### "No microphone detected"
1. Check Windows Sound Settings → Input
2. Make sure your mic is set as default
3. Try a different USB port if external mic

### Dictation is inaccurate
- Use the large model: `python setup.py --large`
- Speak clearly and at a normal pace
- Reduce background noise
- Make sure the mic is close to your mouth

### App freezes on "Loading model"
- The large model can take 10-30 seconds to load — this is normal
- The small model loads in 1-2 seconds

### WSL: "Could not open audio device"
WSL doesn't have direct audio access. You need PulseAudio:
```bash
sudo apt install pulseaudio
# On Windows side, install PulseAudio for Windows
# Then in WSL: export PULSE_SERVER=tcp:$(hostname).local
```
Alternatively, just run it natively on Windows — it's easier.
