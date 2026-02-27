"""
=============================================================================
Basic Speech-to-Text with Tkinter
=============================================================================

HOW IT WORKS (Learning Guide):
------------------------------
This app uses several key technologies working together:

1. AUDIO CAPTURE (PyAudio):
   - Your microphone produces raw audio data as a stream of numbers
   - We capture this in small chunks (4096 frames at a time)
   - Sample rate is 16000 Hz (16,000 samples per second) — this is the
     sweet spot for speech recognition (voice is mostly 300-3400 Hz)

2. SPEECH RECOGNITION (Vosk):
   - Vosk is an OFFLINE speech recognition engine (no internet needed!)
   - It uses a neural network model trained on thousands of hours of speech
   - The model contains: acoustic model (sounds→phonemes), language model
     (phonemes→words), and a dictionary (valid words + pronunciations)
   - It gives us TWO types of results:
     a) "partial" — real-time guesses as you speak (updates constantly)
     b) "final"   — confirmed text after a pause is detected

3. GUI (Tkinter):
   - Simple text editor where recognized speech appears
   - Status bar shows microphone state and partial recognition
   - You can also type normally — it's a full text editor

4. THREADING:
   - Audio capture runs on a SEPARATE THREAD from the GUI
   - Why? The GUI runs an "event loop" — if we blocked it to listen
     for audio, the window would freeze. Threading lets both run at once.
   - We use a Queue (thread-safe) to pass text from the audio thread
     to the GUI thread safely.

5. HOTKEY (pynput):
   - Listens for Ctrl+Shift+D globally (even when app isn't focused)
   - Toggles dictation on/off like Windows' Win+H feature
   - Uses a keyboard listener on yet another thread

ARCHITECTURE DIAGRAM:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Microphone   │────▶│ Audio Thread  │────▶│  Text Queue  │
    │  (PyAudio)    │     │ (Vosk Model)  │     │              │
    └──────────────┘     └──────────────┘     └──────┬───────┘
                                                      │
    ┌──────────────┐     ┌──────────────┐            │
    │ Hotkey Thread │────▶│  GUI Thread   │◀───────────┘
    │  (pynput)     │     │  (Tkinter)    │
    └──────────────┘     └──────────────┘

=============================================================================
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import json
import os
import sys
import time
import math
import wave
import struct

# psutil for resource monitoring (CPU, RAM usage)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: Install 'psutil' for resource monitoring (pip install psutil)")

# ---------------------------------------------------------------------------
# LEARNING NOTE: Error handling for imports
# ---------------------------------------------------------------------------
# We try to import each dependency separately so we can give you a clear
# error message about what's missing, rather than a cryptic traceback.

try:
    import pyaudio
except ImportError:
    print("=" * 60)
    print("MISSING: PyAudio")
    print("Install with: pip install pyaudio")
    print("On Windows, you may need: pip install pipwin && pipwin install pyaudio")
    print("=" * 60)
    sys.exit(1)

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
except ImportError:
    print("=" * 60)
    print("MISSING: Vosk")
    print("Install with: pip install vosk")
    print("=" * 60)
    sys.exit(1)

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    print("=" * 60)
    print("MISSING: pynput")
    print("Install with: pip install pynput")
    print("=" * 60)
    sys.exit(1)


# Suppress Vosk's verbose logging (it prints a LOT by default)
SetLogLevel(-1)


# ===========================================================================
# AUDIO PROCESSOR (Noise Gate + Compressor)
# ===========================================================================
# LEARNING NOTE:
# Audio processing is done in the DIGITAL domain — we're manipulating
# numbers (samples) that represent the audio waveform. Each sample is
# a 16-bit integer (-32768 to 32767) representing the air pressure at
# that instant.
#
# SIGNAL CHAIN (just like a real audio mixer/DAW):
#   Microphone → [Noise Gate] → [Compressor] → Vosk Recognizer
#
# We process audio in "chunks" (buffers of ~4096 samples). For each
# chunk, we calculate its loudness (RMS) and apply the gate/compressor.
# ===========================================================================

class AudioProcessor:
    """
    Real-time audio processor with noise gate and compressor.
    
    NOISE GATE — How it works:
    ─────────────────────────
    A noise gate is like a bouncer at a club — if the audio isn't loud
    enough (below the threshold), it gets silenced. This kills background
    noise like fans, AC, keyboard clicks, etc.
    
    But we can't just hard-cut the audio on/off — that creates ugly clicks
    and pops. So we use "attack" and "release" times to smoothly fade in/out:
    
      Audio Level:  ─────╱██████████╲─────────
      Gate State:   CLOSED │  OPEN   │ CLOSING
                          attack    release
    
    - Threshold: Volume level (in dB) that triggers the gate open
    - Attack:    How fast the gate opens (in seconds) — fast = responsive
    - Hold:      How long to keep gate open after audio drops below threshold
    - Release:   How slowly the gate closes (in seconds) — slow = smooth fade
    
    COMPRESSOR — How it works:
    ──────────────────────────
    A compressor reduces the dynamic range of audio — it makes loud parts
    quieter and (with makeup gain) quiet parts louder. This gives Vosk a
    more consistent signal level to work with.
    
    Think of it like auto-volume: if you shout, it turns you down; if you
    whisper, it turns you up.
    
      Input Level:   ╱╲    ╱────╲      ╱╲
      Output Level:  ╱╲   ╱──╲         ╱╲    (peaks squashed)
    
    - Threshold: Level (in dB) above which compression kicks in
    - Ratio:     How much to reduce signal above threshold
                 (e.g., 4:1 means 4dB of input → 1dB of output above threshold)
    - Attack:    How fast the compressor reacts to loud sounds
    - Release:   How fast the compressor lets go after sound gets quiet
    - Makeup Gain: Volume boost applied after compression to compensate
    """
    
    def __init__(self):
        # ---- Noise Gate Parameters ----
        self.gate_enabled = True
        self.gate_threshold_db = -40.0    # dB — audio below this is silenced
        self.gate_attack = 0.005          # seconds — how fast gate opens (5ms)
        self.gate_hold = 0.15             # seconds — keep open after audio drops
        self.gate_release = 0.05          # seconds — how slowly gate closes (50ms)
        
        # Gate state tracking
        self._gate_gain = 0.0             # Current gate gain (0.0 = closed, 1.0 = open)
        self._gate_hold_counter = 0       # Samples remaining in hold phase
        self._gate_is_open = False
        
        # ---- Compressor Parameters ----
        self.comp_enabled = True
        self.comp_threshold_db = -20.0    # dB — compress audio above this level
        self.comp_ratio = 4.0             # 4:1 ratio — moderate compression
        self.comp_attack = 0.010          # seconds — 10ms attack
        self.comp_release = 0.100         # seconds — 100ms release
        self.comp_makeup_db = 6.0         # dB — boost to compensate for squashing
        
        # Compressor state tracking
        self._comp_envelope = 0.0         # Current envelope follower level
        
        # ---- Audio Format Info ----
        self.sample_rate = 16000
        self.max_amplitude = 32767        # Max value for 16-bit signed int
        
    def _rms_db(self, samples):
        """
        Calculate the RMS (Root Mean Square) level of audio samples in decibels.
        
        LEARNING NOTE:
        RMS is the standard way to measure audio "loudness." It works by:
        1. Square each sample (makes all values positive)
        2. Take the Mean (average) of all squared values
        3. Take the square Root of that mean
        
        Then convert to decibels (dB) because human hearing is logarithmic:
        - 0 dB   = maximum possible level (clipping)
        - -20 dB  = moderate level
        - -40 dB  = quiet
        - -60 dB  = very quiet / near silence
        - -∞ dB   = absolute silence
        
        Formula: dB = 20 * log10(rms / max_amplitude)
        """
        if not samples:
            return -100.0
            
        # Calculate RMS
        sum_squares = sum(s * s for s in samples)
        rms = math.sqrt(sum_squares / len(samples))
        
        # Convert to dB (relative to max amplitude)
        if rms < 1.0:  # Avoid log(0)
            return -100.0
        return 20.0 * math.log10(rms / self.max_amplitude)
    
    def _apply_gate(self, samples):
        """
        Apply noise gate to audio samples.
        
        LEARNING NOTE:
        The gate has 4 states it cycles through:
        
        1. CLOSED (gain = 0): Audio is below threshold, output silence
        2. ATTACK (gain ramping up): Audio crossed threshold, fading in
        3. OPEN (gain = 1): Audio is above threshold, pass through
        4. HOLD → RELEASE (gain ramping down): Audio dropped below threshold,
           wait (hold), then fade out (release)
        
        The "gain" is a multiplier applied to each sample:
        - gain = 0.0 → silence
        - gain = 0.5 → half volume  
        - gain = 1.0 → full volume (pass-through)
        
        We calculate gain change PER SAMPLE for smooth transitions.
        """
        if not self.gate_enabled:
            return samples
            
        rms_db = self._rms_db(samples)
        
        # Calculate per-sample gain change rates
        # (how much the gain changes for each individual sample)
        attack_rate = 1.0 / max(1, self.gate_attack * self.sample_rate)
        release_rate = 1.0 / max(1, self.gate_release * self.sample_rate)
        hold_samples = int(self.gate_hold * self.sample_rate)
        
        output = []
        
        if rms_db >= self.gate_threshold_db:
            # Audio is above threshold — gate should be OPEN
            self._gate_is_open = True
            self._gate_hold_counter = hold_samples  # Reset hold timer
            
            for s in samples:
                # Ramp gain up (attack phase)
                self._gate_gain = min(1.0, self._gate_gain + attack_rate)
                output.append(int(s * self._gate_gain))
        else:
            # Audio is below threshold
            if self._gate_hold_counter > 0:
                # Still in HOLD phase — keep gate open
                for s in samples:
                    self._gate_hold_counter -= 1
                    output.append(int(s * self._gate_gain))
            else:
                # RELEASE phase — ramp gain down
                self._gate_is_open = False
                for s in samples:
                    self._gate_gain = max(0.0, self._gate_gain - release_rate)
                    output.append(int(s * self._gate_gain))
                    
        return output
    
    def _apply_compressor(self, samples):
        """
        Apply dynamic range compression to audio samples.
        
        LEARNING NOTE:
        Compression is the most important tool in professional audio.
        Here's the math:
        
        1. For each sample, calculate its level in dB
        2. If it's above the threshold, calculate how much it exceeds it
        3. Reduce the excess by the ratio (e.g., 4:1 means 4dB over → 1dB over)
        4. Apply makeup gain to bring the overall level back up
        
        Example with threshold=-20dB, ratio=4:1, makeup=+6dB:
          Input: -10dB (10dB over threshold)
          Compressed excess: 10dB / 4 = 2.5dB over threshold
          Output before makeup: -20 + 2.5 = -17.5dB  
          Output after makeup: -17.5 + 6 = -11.5dB
        
        The envelope follower smooths the gain changes using attack/release
        times, preventing distortion from sudden gain changes.
        """
        if not self.comp_enabled:
            return samples
            
        # Per-sample smoothing coefficients
        # These control how fast the compressor reacts
        attack_coeff = math.exp(-1.0 / max(1, self.comp_attack * self.sample_rate))
        release_coeff = math.exp(-1.0 / max(1, self.comp_release * self.sample_rate))
        
        # Makeup gain as a linear multiplier
        makeup_linear = 10.0 ** (self.comp_makeup_db / 20.0)
        
        output = []
        
        for s in samples:
            # Get absolute value for level detection
            abs_s = abs(s) / self.max_amplitude  # Normalize to 0.0-1.0
            
            # Envelope follower — smoothly tracks the audio level
            if abs_s > self._comp_envelope:
                # Attack: level is rising, follow quickly
                self._comp_envelope = attack_coeff * self._comp_envelope + (1 - attack_coeff) * abs_s
            else:
                # Release: level is falling, follow slowly  
                self._comp_envelope = release_coeff * self._comp_envelope + (1 - release_coeff) * abs_s
            
            # Convert envelope to dB
            if self._comp_envelope < 0.00001:  # Near silence
                output.append(s)
                continue
                
            env_db = 20.0 * math.log10(self._comp_envelope)
            
            # Calculate gain reduction
            if env_db > self.comp_threshold_db:
                # Above threshold — apply compression
                over_db = env_db - self.comp_threshold_db
                compressed_over = over_db / self.comp_ratio
                gain_reduction_db = over_db - compressed_over  # How much to reduce
                gain_linear = 10.0 ** (-gain_reduction_db / 20.0)
            else:
                # Below threshold — no compression
                gain_linear = 1.0
            
            # Apply gain reduction + makeup gain
            processed = s * gain_linear * makeup_linear
            
            # Clip to 16-bit range to prevent distortion
            processed = max(-32768, min(32767, int(processed)))
            output.append(processed)
            
        return output
    
    def process(self, raw_bytes):
        """
        Process a chunk of raw audio bytes through the signal chain.
        
        LEARNING NOTE:
        Audio comes from PyAudio as raw bytes. We need to:
        1. UNPACK bytes → list of integers (samples)
        2. Process the samples (gate → compressor)
        3. REPACK integers → bytes for Vosk
        
        struct.unpack('<Nh', data) reads N little-endian 16-bit signed ints
        struct.pack('<Nh', *samples) writes them back to bytes
        
        The '<' means little-endian (least significant byte first), which is
        the standard byte order on x86/x64 processors (Windows & Linux PCs).
        'h' means signed 16-bit integer (short).
        """
        # Unpack raw bytes into integer samples
        num_samples = len(raw_bytes) // 2  # 2 bytes per 16-bit sample
        samples = list(struct.unpack(f'<{num_samples}h', raw_bytes))
        
        # Calculate input level for metering (before processing)
        self.input_level_db = self._rms_db(samples)
        
        # Signal chain: Gate → Compressor
        samples = self._apply_gate(samples)
        samples = self._apply_compressor(samples)
        
        # Calculate output level for metering (after processing)
        self.output_level_db = self._rms_db(samples)
        
        # Pack back to bytes
        return struct.pack(f'<{num_samples}h', *samples)
    
    @property
    def gate_is_open(self):
        """Whether the noise gate is currently passing audio."""
        return self._gate_is_open


# ===========================================================================
# AUDIO ENGINE
# ===========================================================================
# LEARNING NOTE:
# This class handles all the microphone + Vosk interaction.
# It runs on its own thread so the GUI stays responsive.
# The "callback" pattern lets us notify the GUI without tight coupling.
# ===========================================================================

class AudioEngine:
    """
    Captures microphone audio and runs it through Vosk for recognition.
    
    Key concepts:
    - Sample Rate: How many audio samples per second (16000 is standard for speech)
    - Chunk Size: How many samples we process at once (4096 ≈ 0.25 seconds of audio)
    - Channels: 1 = mono (single microphone), which is all we need for speech
    - Format: paInt16 = 16-bit integers, the standard for speech audio
    """
    
    SAMPLE_RATE = 16000      # Hz — optimized for human voice frequency range
    CHUNK_SIZE = 4096        # frames per buffer — balance between latency and stability
    CHANNELS = 1             # mono audio (speech recognition doesn't need stereo)
    FORMAT = pyaudio.paInt16  # 16-bit audio (65,536 possible amplitude values)
    
    def __init__(self, model_path, on_partial=None, on_final=None, on_error=None, device_index=None):
        """
        Args:
            model_path: Path to the Vosk language model directory
            on_partial: Callback function for partial (in-progress) results
            on_final: Callback function for final (confirmed) results  
            on_error: Callback function for error reporting
            device_index: PyAudio device index for mic (None = system default)
        """
        self.model_path = model_path
        self.on_partial = on_partial or (lambda text: None)
        self.on_final = on_final or (lambda text: None)
        self.on_error = on_error or (lambda err: None)
        self.device_index = device_index
        
        self.model = None
        self.recognizer = None
        self.audio = None
        self.stream = None
        self.is_listening = False
        self._thread = None
        
        # Audio processor (noise gate + compressor)
        self.processor = AudioProcessor()
        
    def load_model(self):
        """
        Load the Vosk language model.
        
        LEARNING NOTE:
        The model is a directory containing several files:
        - am/ (acoustic model) — neural network that converts audio→phonemes
        - graph/ (language model) — statistical model of word sequences
        - conf/  — configuration files
        
        Larger models = more accurate but use more RAM and are slower to load.
        The "small" English model is ~50MB and works great for dictation.
        The "large" model is ~1.8GB and is near-professional quality.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at: {self.model_path}\n"
                f"Download from: https://alphacephei.com/vosk/models\n"
                f"Recommended: vosk-model-small-en-us-0.15 (fast, ~50MB)\n"
                f"          or: vosk-model-en-us-0.22 (accurate, ~1.8GB)"
            )
        
        print(f"Loading model from {self.model_path}...")
        self.model = Model(self.model_path)
        
        # The KaldiRecognizer is the actual recognition engine.
        # It takes the model and the sample rate so it knows how to
        # interpret the raw audio bytes it receives.
        self.recognizer = KaldiRecognizer(self.model, self.SAMPLE_RATE)
        
        # Enable word-level timestamps in results (useful for future features
        # like highlighting words as they're spoken)
        self.recognizer.SetWords(True)
        
        # Enable partial results so we can show real-time feedback
        self.recognizer.SetPartialWords(True)
        
        print("Model loaded successfully!")
        
    def start(self):
        """Start listening on a background thread."""
        if self.model is None:
            raise RuntimeError("Model not loaded! Call load_model() first.")
        if self.is_listening:
            return
            
        self.is_listening = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop listening and clean up audio resources."""
        self.is_listening = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cleanup_stream()
        
    def _cleanup_stream(self):
        """Safely close the audio stream and PyAudio instance."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None
            
    def _listen_loop(self):
        """
        Main audio capture loop (runs on background thread).
        
        LEARNING NOTE:
        This is the heart of the app. Here's what happens each iteration:
        
        1. Read a chunk of audio from the microphone (blocking call)
        2. Feed it to the Vosk recognizer
        3. Check if Vosk has a partial result (still processing)
           or a final result (detected end of utterance)
        4. Call the appropriate callback with the recognized text
        
        The recognizer.AcceptWaveform() method returns:
        - True  → it has a final result (speaker paused or sentence ended)
        - False → it has a partial result (still listening, best guess so far)
        """
        try:
            self.audio = pyaudio.PyAudio()
            
            # Build stream kwargs — only include input_device_index if specified
            stream_kwargs = dict(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
            )
            if self.device_index is not None:
                stream_kwargs["input_device_index"] = self.device_index
            
            self.stream = self.audio.open(**stream_kwargs)
            self.stream.start_stream()
            
            while self.is_listening:
                try:
                    # Read audio data from microphone
                    # This blocks until CHUNK_SIZE frames are available
                    data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    
                    # Bail out if we were told to stop while reading
                    if not self.is_listening:
                        break
                    
                    # Process audio through noise gate and compressor
                    # LEARNING NOTE: This is the signal chain —
                    # raw mic → gate (kill noise) → compressor (even out volume) → Vosk
                    data = self.processor.process(data)
                    
                    # Feed processed audio to Vosk recognizer
                    if self.recognizer.AcceptWaveform(data):
                        # FINAL result — Vosk detected end of utterance
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            self.on_final(text)
                    else:
                        # PARTIAL result — still listening, showing best guess
                        partial = json.loads(self.recognizer.PartialResult())
                        text = partial.get("partial", "").strip()
                        if text:
                            self.on_partial(text)
                            
                except OSError as e:
                    # Audio buffer overflow — not critical, just skip this chunk
                    if "Input overflowed" in str(e):
                        continue
                    # If we're shutting down, this is expected — don't report it
                    if not self.is_listening:
                        break
                    raise
                except Exception:
                    # If we're shutting down, swallow errors from the stream closing
                    if not self.is_listening:
                        break
                    raise
                    
        except Exception as e:
            # Only report errors if we weren't intentionally stopped
            if self.is_listening:
                self.on_error(str(e))
        finally:
            self._cleanup_stream()
    
    def get_final_result(self):
        """
        Flush any remaining audio in the recognizer's buffer.
        Call this when stopping dictation to get any last words.
        """
        if self.recognizer:
            result = json.loads(self.recognizer.FinalResult())
            text = result.get("text", "").strip()
            if text:
                self.on_final(text)


# ===========================================================================
# PUNCTUATION & TEXT FORMATTING
# ===========================================================================
# LEARNING NOTE:
# Speech recognition gives us raw words. But for dictation, people want to
# say "period" and get "." — just like Windows dictation. This processor
# handles those voice commands.
# ===========================================================================

class TextProcessor:
    """
    Converts spoken commands into punctuation and formatting.
    
    Say "hello period new line how are you question mark" and get:
    "Hello.
    How are you?"
    
    This is how professional dictation software works — the user speaks
    punctuation and formatting commands as words.
    """
    
    # Map of spoken words → punctuation characters
    PUNCTUATION = {
        "period": ".",
        "full stop": ".",
        "comma": ",",
        "question mark": "?",
        "exclamation mark": "!",
        "exclamation point": "!",
        "colon": ":",
        "semicolon": ";",
        "dash": "—",
        "hyphen": "-",
        "open parenthesis": "(",
        "close parenthesis": ")",
        "open bracket": "[",
        "close bracket": "]",
        "open quote": '"',
        "close quote": '"',
        "single quote": "'",
        "apostrophe": "'",
        "ellipsis": "...",
        "at sign": "@",
        "hashtag": "#",
        "dollar sign": "$",
        "percent": "%",
        "ampersand": "&",
        "asterisk": "*",
        "slash": "/",
        "backslash": "\\",
        "underscore": "_",
        "equals": "=",
        "plus sign": "+",
    }
    
    # Formatting commands
    COMMANDS = {
        "new line": "\n",
        "newline": "\n",
        "new paragraph": "\n\n",
        "tab": "\t",
    }
    
    # Words that should be capitalized (next word)
    CAPITALIZE_AFTER = {".", "!", "?", "\n", "\n\n"}
    
    def __init__(self):
        self.capitalize_next = True  # Start of text = capitalize first word
        self._buffer = ""
        
    def process(self, text):
        """
        Process raw recognized text into formatted text with punctuation.
        
        LEARNING NOTE:
        We scan through the words looking for punctuation commands.
        Some commands are TWO words (like "question mark"), so we use
        a lookahead approach — check if current + next word form a command.
        
        Args:
            text: Raw text from speech recognizer (lowercase, no punctuation)
            
        Returns:
            Formatted text with punctuation and proper capitalization
        """
        if not text:
            return ""
            
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            # Check for two-word commands first (e.g., "question mark")
            if i + 1 < len(words):
                two_words = f"{words[i]} {words[i+1]}"
                
                if two_words in self.PUNCTUATION:
                    punct = self.PUNCTUATION[two_words]
                    # Punctuation attaches to previous word (no space before)
                    if result and result[-1] != " ":
                        result.append(punct)
                    else:
                        # Remove trailing space before punctuation
                        if result and result[-1] == " ":
                            result.pop()
                        result.append(punct)
                    
                    if punct in self.CAPITALIZE_AFTER:
                        self.capitalize_next = True
                    result.append(" ")
                    i += 2
                    continue
                    
                if two_words in self.COMMANDS:
                    cmd = self.COMMANDS[two_words]
                    # Remove trailing space before command
                    if result and result[-1] == " ":
                        result.pop()
                    result.append(cmd)
                    self.capitalize_next = True
                    i += 2
                    continue
            
            # Check for single-word commands
            word = words[i]
            
            if word in self.PUNCTUATION:
                punct = self.PUNCTUATION[word]
                if result and result[-1] == " ":
                    result.pop()
                result.append(punct)
                if punct in self.CAPITALIZE_AFTER:
                    self.capitalize_next = True
                result.append(" ")
                i += 1
                continue
                
            if word in self.COMMANDS:
                cmd = self.COMMANDS[word]
                if result and result[-1] == " ":
                    result.pop()
                result.append(cmd)
                self.capitalize_next = True
                i += 1
                continue
            
            # Regular word — apply capitalization if needed
            if self.capitalize_next:
                word = word.capitalize()
                self.capitalize_next = False
                
            result.append(word)
            result.append(" ")
            i += 1
        
        output = "".join(result)
        # Remove trailing space (but keep trailing newlines)
        if output.endswith(" "):
            output = output[:-1]
            
        return output
        
    def reset(self):
        """Reset capitalization state (e.g., when starting fresh)."""
        self.capitalize_next = True


# ===========================================================================
# MAIN APPLICATION (GUI)
# ===========================================================================
# LEARNING NOTE:
# Tkinter is Python's built-in GUI library. Every Tkinter app has:
# 1. A root window (Tk())
# 2. Widgets placed inside it (buttons, text areas, labels, etc.)
# 3. An event loop (mainloop()) that waits for user interaction
#
# We use `after()` to schedule periodic checks of our text queue,
# which is how the audio thread communicates with the GUI thread safely.
# ===========================================================================

class SpeechToTextApp:
    """Main application window and controller."""
    
    # Default model path — user can change this
    DEFAULT_MODEL_DIR = "vosk-model"
    
    def __init__(self):
        # ---- State ----
        self.is_dictating = False
        self.text_queue = queue.Queue()      # Thread-safe queue for recognized text
        self.partial_queue = queue.Queue()    # Queue for partial (in-progress) text
        self.engine = None
        self.processor = TextProcessor()
        self.model_path = self.DEFAULT_MODEL_DIR
        self._selected_mic_index = None  # None = use system default
        
        # ---- Build GUI ----
        self.root = tk.Tk()
        self.root.title("Basic Speech-to-Text with Tkinter")
        self.root.geometry("950x700")
        self.root.minsize(700, 500)
        
        # Set icon and styling
        self.root.configure(bg="#1e1e2e")
        self._setup_styles()
        self._build_toolbar()
        self._build_audio_controls()
        self._build_text_area()
        self._build_partial_display()
        self._build_status_bar()
        
        # ---- Keyboard Shortcuts ----
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<F1>", self._toggle_commands_panel)
        
        # ---- Global Hotkey (works even when window isn't focused) ----
        self._setup_global_hotkey()
        
        # ---- Start polling the text queue ----
        self._poll_queues()
        
        # ---- Handle window close ----
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _setup_styles(self):
        """Configure ttk styles for a modern dark theme."""
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Color palette (Catppuccin Mocha inspired)
        self.colors = {
            "bg":       "#1e1e2e",
            "surface":  "#313244",
            "text":     "#cdd6f4",
            "subtext":  "#a6adc8",
            "green":    "#a6e3a1",
            "red":      "#f38ba8",
            "blue":     "#89b4fa",
            "yellow":   "#f9e2af",
            "accent":   "#cba6f7",
        }
        
        self.style.configure("Toolbar.TFrame", background=self.colors["surface"])
        self.style.configure("Status.TFrame", background=self.colors["surface"])
        self.style.configure("Status.TLabel",
                             background=self.colors["surface"],
                             foreground=self.colors["subtext"],
                             font=("Consolas", 10))
        
        # Button styles
        self.style.configure("Action.TButton",
                             background=self.colors["blue"],
                             foreground=self.colors["bg"],
                             font=("Segoe UI", 10, "bold"),
                             padding=(12, 6))
        self.style.map("Action.TButton",
                       background=[("active", self.colors["accent"])])
        
        self.style.configure("Dictate.TButton",
                             background=self.colors["green"],
                             foreground=self.colors["bg"],
                             font=("Segoe UI", 11, "bold"),
                             padding=(16, 8))
        self.style.map("Dictate.TButton",
                       background=[("active", self.colors["yellow"])])
        
    def _build_toolbar(self):
        """Build the top toolbar with action buttons."""
        toolbar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=5)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        
        # File operations
        ttk.Button(toolbar, text="New", style="Action.TButton",
                   command=self.new_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open", style="Action.TButton",
                   command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", style="Action.TButton",
                   command=self.save_file).pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                         padx=10, pady=2)
        
        # Model path selector
        ttk.Label(toolbar, text="Model:",
                  background=self.colors["surface"],
                  foreground=self.colors["subtext"],
                  font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(5, 2))
        
        self.model_var = tk.StringVar(value=self.model_path)
        model_entry = ttk.Entry(toolbar, textvariable=self.model_var, width=30)
        model_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Browse", style="Action.TButton",
                   command=self._browse_model).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Load Model", style="Action.TButton",
                   command=self._load_model_async).pack(side=tk.LEFT, padx=2)
        
        # Dictation toggle (prominent button on the right)
        self.dictate_btn = ttk.Button(toolbar, text="🎤 Start Dictation",
                                       style="Dictate.TButton",
                                       command=self.toggle_dictation)
        self.dictate_btn.pack(side=tk.RIGHT, padx=5)
        
        # Commands reference toggle
        self.help_btn = tk.Button(
            toolbar, text="❓ Commands",
            font=("Segoe UI", 10),
            bg=self.colors["accent"],
            fg=self.colors["bg"],
            activebackground=self.colors["blue"],
            activeforeground=self.colors["bg"],
            bd=0, padx=10, pady=5,
            cursor="hand2",
            command=self._toggle_commands_panel,
        )
        self.help_btn.pack(side=tk.RIGHT, padx=2)
        
    def _build_audio_controls(self):
        """
        Build the collapsible audio processing controls panel.
        
        LEARNING NOTE:
        This panel lets you adjust the noise gate and compressor in real-time.
        We use Tkinter Scale widgets (sliders) that call callback functions
        whenever their value changes. The callbacks update the AudioProcessor
        parameters on the fly — no need to restart dictation.
        
        The panel is collapsible (toggle show/hide) so it doesn't clutter
        the interface when you don't need it.
        """
        # Toggle button in its own frame
        toggle_frame = tk.Frame(self.root, bg=self.colors["surface"])
        toggle_frame.pack(fill=tk.X, side=tk.TOP)
        
        self._audio_panel_visible = False
        self.audio_toggle_btn = tk.Button(
            toggle_frame,
            text="▶ Audio Processing",
            font=("Segoe UI", 9),
            bg=self.colors["surface"],
            fg=self.colors["subtext"],
            activebackground=self.colors["surface"],
            activeforeground=self.colors["text"],
            bd=0,
            cursor="hand2",
            command=self._toggle_audio_panel,
        )
        self.audio_toggle_btn.pack(side=tk.LEFT, padx=10, pady=2)
        
        # Level meter labels (always visible next to toggle)
        self.input_level_label = tk.Label(
            toggle_frame, text="IN: ---  ",
            font=("Consolas", 9), bg=self.colors["surface"],
            fg=self.colors["green"], anchor=tk.E,
        )
        self.input_level_label.pack(side=tk.RIGHT, padx=5)
        
        self.output_level_label = tk.Label(
            toggle_frame, text="OUT: ---  ",
            font=("Consolas", 9), bg=self.colors["surface"],
            fg=self.colors["blue"], anchor=tk.E,
        )
        self.output_level_label.pack(side=tk.RIGHT, padx=5)
        
        # Collapsible panel
        self.audio_panel = tk.Frame(self.root, bg=self.colors["surface"], padx=10, pady=5)
        # Don't pack yet — starts hidden
        
        # ---- Noise Gate Controls ----
        gate_frame = tk.LabelFrame(
            self.audio_panel, text=" Noise Gate ",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["surface"], fg=self.colors["green"],
            padx=10, pady=5,
        )
        gate_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Gate enable checkbox
        self.gate_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            gate_frame, text="Enabled", variable=self.gate_enabled_var,
            bg=self.colors["surface"], fg=self.colors["text"],
            selectcolor=self.colors["bg"], activebackground=self.colors["surface"],
            activeforeground=self.colors["text"],
            command=self._update_gate_enabled,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Gate threshold slider
        tk.Label(gate_frame, text="Threshold (dB):",
                 bg=self.colors["surface"], fg=self.colors["subtext"],
                 font=("Segoe UI", 9)).grid(row=1, column=0, sticky=tk.W)
        self.gate_thresh_var = tk.DoubleVar(value=-26.0)
        self.gate_thresh_slider = tk.Scale(
            gate_frame, from_=-60, to=0, orient=tk.HORIZONTAL,
            variable=self.gate_thresh_var, resolution=1,
            bg=self.colors["surface"], fg=self.colors["text"],
            troughcolor=self.colors["bg"], highlightthickness=0,
            length=200, command=self._update_gate_threshold,
        )
        self.gate_thresh_slider.grid(row=1, column=1, sticky=tk.EW, padx=5)
        
        # Gate hold slider
        tk.Label(gate_frame, text="Hold (ms):",
                 bg=self.colors["surface"], fg=self.colors["subtext"],
                 font=("Segoe UI", 9)).grid(row=2, column=0, sticky=tk.W)
        self.gate_hold_var = tk.DoubleVar(value=150.0)
        self.gate_hold_slider = tk.Scale(
            gate_frame, from_=0, to=500, orient=tk.HORIZONTAL,
            variable=self.gate_hold_var, resolution=10,
            bg=self.colors["surface"], fg=self.colors["text"],
            troughcolor=self.colors["bg"], highlightthickness=0,
            length=200, command=self._update_gate_hold,
        )
        self.gate_hold_slider.grid(row=2, column=1, sticky=tk.EW, padx=5)
        
        # Gate release slider
        tk.Label(gate_frame, text="Release (ms):",
                 bg=self.colors["surface"], fg=self.colors["subtext"],
                 font=("Segoe UI", 9)).grid(row=3, column=0, sticky=tk.W)
        self.gate_release_var = tk.DoubleVar(value=100.0)
        self.gate_release_slider = tk.Scale(
            gate_frame, from_=10, to=500, orient=tk.HORIZONTAL,
            variable=self.gate_release_var, resolution=10,
            bg=self.colors["surface"], fg=self.colors["text"],
            troughcolor=self.colors["bg"], highlightthickness=0,
            length=200, command=self._update_gate_release,
        )
        self.gate_release_slider.grid(row=3, column=1, sticky=tk.EW, padx=5)
        
        gate_frame.columnconfigure(1, weight=1)
        
        # ---- Compressor Controls ----
        comp_frame = tk.LabelFrame(
            self.audio_panel, text=" Compressor ",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["surface"], fg=self.colors["blue"],
            padx=10, pady=5,
        )
        comp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Compressor enable checkbox
        self.comp_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            comp_frame, text="Enabled", variable=self.comp_enabled_var,
            bg=self.colors["surface"], fg=self.colors["text"],
            selectcolor=self.colors["bg"], activebackground=self.colors["surface"],
            activeforeground=self.colors["text"],
            command=self._update_comp_enabled,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Compressor threshold slider
        tk.Label(comp_frame, text="Threshold (dB):",
                 bg=self.colors["surface"], fg=self.colors["subtext"],
                 font=("Segoe UI", 9)).grid(row=1, column=0, sticky=tk.W)
        self.comp_thresh_var = tk.DoubleVar(value=-16.0)
        self.comp_thresh_slider = tk.Scale(
            comp_frame, from_=-40, to=0, orient=tk.HORIZONTAL,
            variable=self.comp_thresh_var, resolution=1,
            bg=self.colors["surface"], fg=self.colors["text"],
            troughcolor=self.colors["bg"], highlightthickness=0,
            length=200, command=self._update_comp_threshold,
        )
        self.comp_thresh_slider.grid(row=1, column=1, sticky=tk.EW, padx=5)
        
        # Compressor ratio slider
        tk.Label(comp_frame, text="Ratio:",
                 bg=self.colors["surface"], fg=self.colors["subtext"],
                 font=("Segoe UI", 9)).grid(row=2, column=0, sticky=tk.W)
        self.comp_ratio_var = tk.DoubleVar(value=4.0)
        self.comp_ratio_slider = tk.Scale(
            comp_frame, from_=1, to=20, orient=tk.HORIZONTAL,
            variable=self.comp_ratio_var, resolution=0.5,
            bg=self.colors["surface"], fg=self.colors["text"],
            troughcolor=self.colors["bg"], highlightthickness=0,
            length=200, command=self._update_comp_ratio,
        )
        self.comp_ratio_slider.grid(row=2, column=1, sticky=tk.EW, padx=5)
        
        # Makeup gain slider
        tk.Label(comp_frame, text="Makeup Gain (dB):",
                 bg=self.colors["surface"], fg=self.colors["subtext"],
                 font=("Segoe UI", 9)).grid(row=3, column=0, sticky=tk.W)
        self.comp_makeup_var = tk.DoubleVar(value=6.0)
        self.comp_makeup_slider = tk.Scale(
            comp_frame, from_=0, to=24, orient=tk.HORIZONTAL,
            variable=self.comp_makeup_var, resolution=1,
            bg=self.colors["surface"], fg=self.colors["text"],
            troughcolor=self.colors["bg"], highlightthickness=0,
            length=200, command=self._update_comp_makeup,
        )
        self.comp_makeup_slider.grid(row=3, column=1, sticky=tk.EW, padx=5)
        
        comp_frame.columnconfigure(1, weight=1)
        
    def _toggle_audio_panel(self):
        """Show/hide the audio processing controls panel."""
        if self._audio_panel_visible:
            self.audio_panel.pack_forget()
            self.audio_toggle_btn.config(text="▶ Audio Processing")
            self._audio_panel_visible = False
        else:
            # Pack it right after the toggle button's parent frame
            self.audio_panel.pack(fill=tk.X, side=tk.TOP, before=self.content_frame)
            self.audio_toggle_btn.config(text="▼ Audio Processing")
            self._audio_panel_visible = True
    
    # ---- Audio Parameter Update Callbacks ----
    # LEARNING NOTE:
    # Tkinter Scale widgets pass the new value as a string argument
    # to the callback. We update the AudioProcessor parameters directly.
    # Because the audio thread only READS these values (never writes),
    # this is thread-safe without locks for simple float assignments.
    
    def _get_audio_processor(self):
        """Safely get the audio processor from the engine, if loaded."""
        if self.engine and hasattr(self.engine, 'processor'):
            return self.engine.processor
        return None
    
    def _update_gate_enabled(self):
        proc = self._get_audio_processor()
        if proc:
            proc.gate_enabled = self.gate_enabled_var.get()
    
    def _update_gate_threshold(self, val):
        proc = self._get_audio_processor()
        if proc:
            proc.gate_threshold_db = float(val)
    
    def _update_gate_hold(self, val):
        proc = self._get_audio_processor()
        if proc:
            proc.gate_hold = float(val) / 1000.0  # Convert ms to seconds
    
    def _update_gate_release(self, val):
        proc = self._get_audio_processor()
        if proc:
            proc.gate_release = float(val) / 1000.0
    
    def _update_comp_enabled(self):
        proc = self._get_audio_processor()
        if proc:
            proc.comp_enabled = self.comp_enabled_var.get()
    
    def _update_comp_threshold(self, val):
        proc = self._get_audio_processor()
        if proc:
            proc.comp_threshold_db = float(val)
    
    def _update_comp_ratio(self, val):
        proc = self._get_audio_processor()
        if proc:
            proc.comp_ratio = float(val)
    
    def _update_comp_makeup(self, val):
        proc = self._get_audio_processor()
        if proc:
            proc.update_makeup_gain(float(val))

    def _build_audio_controls(self):
        """
        Build the collapsible audio processing controls panel.
        
        LEARNING NOTE:
        This panel gives you real-time control over the noise gate and
        compressor, just like a mixing console or DAW. The level meters
        show you the input/output levels so you can see the processing
        in action.
        
        Each slider updates the AudioProcessor parameters in real-time
        via Tkinter's variable tracing (the 'command' callback on Scale).
        """
        # ---- Collapsible container ----
        self._audio_panel_visible = tk.BooleanVar(value=False)
        
        toggle_frame = tk.Frame(self.root, bg=self.colors["surface"])
        toggle_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        self._audio_toggle_btn = tk.Button(
            toggle_frame,
            text="▶ Audio Processing",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["subtext"],
            activebackground=self.colors["surface"],
            activeforeground=self.colors["text"],
            bd=0,
            cursor="hand2",
            command=self._toggle_audio_panel,
        )
        self._audio_toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Gate status indicator
        self.gate_status_label = tk.Label(
            toggle_frame,
            text="Gate: ─",
            font=("Consolas", 9),
            bg=self.colors["surface"],
            fg=self.colors["subtext"],
        )
        self.gate_status_label.pack(side=tk.RIGHT, padx=10)
        
        # Level meter (always visible, compact)
        meter_frame = tk.Frame(toggle_frame, bg=self.colors["surface"])
        meter_frame.pack(side=tk.RIGHT, padx=5)
        
        tk.Label(meter_frame, text="IN", font=("Consolas", 8),
                 bg=self.colors["surface"], fg=self.colors["subtext"]
                 ).pack(side=tk.LEFT, padx=(0, 2))
        self.input_meter = tk.Canvas(meter_frame, width=100, height=10,
                                      bg="#11111b", highlightthickness=0)
        self.input_meter.pack(side=tk.LEFT, padx=2)
        
        tk.Label(meter_frame, text="OUT", font=("Consolas", 8),
                 bg=self.colors["surface"], fg=self.colors["subtext"]
                 ).pack(side=tk.LEFT, padx=(8, 2))
        self.output_meter = tk.Canvas(meter_frame, width=100, height=10,
                                       bg="#11111b", highlightthickness=0)
        self.output_meter.pack(side=tk.LEFT, padx=2)
        
        # ---- Expandable controls panel ----
        self._audio_panel = tk.Frame(self.root, bg=self.colors["bg"])
        # Don't pack yet — starts collapsed
        
        # Slider styling
        slider_opts = dict(
            orient=tk.HORIZONTAL,
            length=180,
            bg=self.colors["bg"],
            fg=self.colors["text"],
            troughcolor=self.colors["surface"],
            highlightthickness=0,
            font=("Consolas", 9),
        )
        
        # ---- Noise Gate Controls ----
        gate_frame = tk.LabelFrame(
            self._audio_panel,
            text=" Noise Gate ",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["green"],
            padx=10, pady=5,
        )
        gate_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 2), pady=5)
        
        # Gate enable toggle
        self._gate_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            gate_frame, text="Enabled", variable=self._gate_enabled_var,
            bg=self.colors["bg"], fg=self.colors["text"],
            selectcolor=self.colors["surface"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["text"],
            font=("Segoe UI", 9),
            command=self._update_gate_params,
        ).pack(anchor=tk.W)
        
        # Gate Threshold
        tk.Label(gate_frame, text="Threshold (dB)", bg=self.colors["bg"],
                 fg=self.colors["subtext"], font=("Consolas", 9)).pack(anchor=tk.W)
        self._gate_thresh_var = tk.DoubleVar(value=-40.0)
        tk.Scale(gate_frame, variable=self._gate_thresh_var,
                 from_=-80, to=0, resolution=1,
                 command=lambda _: self._update_gate_params(),
                 **slider_opts).pack(fill=tk.X)
        
        # Gate Hold
        tk.Label(gate_frame, text="Hold (ms)", bg=self.colors["bg"],
                 fg=self.colors["subtext"], font=("Consolas", 9)).pack(anchor=tk.W)
        self._gate_hold_var = tk.DoubleVar(value=150.0)
        tk.Scale(gate_frame, variable=self._gate_hold_var,
                 from_=0, to=500, resolution=10,
                 command=lambda _: self._update_gate_params(),
                 **slider_opts).pack(fill=tk.X)
        
        # Gate Release
        tk.Label(gate_frame, text="Release (ms)", bg=self.colors["bg"],
                 fg=self.colors["subtext"], font=("Consolas", 9)).pack(anchor=tk.W)
        self._gate_release_var = tk.DoubleVar(value=50.0)
        tk.Scale(gate_frame, variable=self._gate_release_var,
                 from_=5, to=200, resolution=5,
                 command=lambda _: self._update_gate_params(),
                 **slider_opts).pack(fill=tk.X)
        
        # ---- Compressor Controls ----
        comp_frame = tk.LabelFrame(
            self._audio_panel,
            text=" Compressor ",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["accent"],
            padx=10, pady=5,
        )
        comp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 5), pady=5)
        
        # Compressor enable toggle
        self._comp_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            comp_frame, text="Enabled", variable=self._comp_enabled_var,
            bg=self.colors["bg"], fg=self.colors["text"],
            selectcolor=self.colors["surface"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["text"],
            font=("Segoe UI", 9),
            command=self._update_comp_params,
        ).pack(anchor=tk.W)
        
        # Compressor Threshold
        tk.Label(comp_frame, text="Threshold (dB)", bg=self.colors["bg"],
                 fg=self.colors["subtext"], font=("Consolas", 9)).pack(anchor=tk.W)
        self._comp_thresh_var = tk.DoubleVar(value=-20.0)
        tk.Scale(comp_frame, variable=self._comp_thresh_var,
                 from_=-60, to=0, resolution=1,
                 command=lambda _: self._update_comp_params(),
                 **slider_opts).pack(fill=tk.X)
        
        # Compressor Ratio
        tk.Label(comp_frame, text="Ratio (N:1)", bg=self.colors["bg"],
                 fg=self.colors["subtext"], font=("Consolas", 9)).pack(anchor=tk.W)
        self._comp_ratio_var = tk.DoubleVar(value=4.0)
        tk.Scale(comp_frame, variable=self._comp_ratio_var,
                 from_=1, to=20, resolution=0.5,
                 command=lambda _: self._update_comp_params(),
                 **slider_opts).pack(fill=tk.X)
        
        # Makeup Gain
        tk.Label(comp_frame, text="Makeup Gain (dB)", bg=self.colors["bg"],
                 fg=self.colors["subtext"], font=("Consolas", 9)).pack(anchor=tk.W)
        self._comp_makeup_var = tk.DoubleVar(value=6.0)
        tk.Scale(comp_frame, variable=self._comp_makeup_var,
                 from_=0, to=24, resolution=1,
                 command=lambda _: self._update_comp_params(),
                 **slider_opts).pack(fill=tk.X)
        
    def _toggle_audio_panel(self):
        """Show/hide the audio processing controls."""
        if self._audio_panel_visible.get():
            self._audio_panel.pack_forget()
            self._audio_toggle_btn.config(text="▶ Audio Processing")
            self._audio_panel_visible.set(False)
        else:
            # Pack it right after the toggle button's parent
            self._audio_panel.pack(fill=tk.X, padx=5, after=self._audio_toggle_btn.master)
            self._audio_toggle_btn.config(text="▼ Audio Processing")
            self._audio_panel_visible.set(True)
            
    def _update_gate_params(self):
        """Push GUI slider values to the AudioProcessor noise gate."""
        if self.engine and self.engine.processor:
            proc = self.engine.processor
            proc.gate_enabled = self._gate_enabled_var.get()
            proc.gate_threshold_db = self._gate_thresh_var.get()
            proc.gate_hold = self._gate_hold_var.get() / 1000.0      # ms → seconds
            proc.gate_release = self._gate_release_var.get() / 1000.0  # ms → seconds
            
    def _update_comp_params(self):
        """Push GUI slider values to the AudioProcessor compressor."""
        if self.engine and self.engine.processor:
            proc = self.engine.processor
            proc.comp_enabled = self._comp_enabled_var.get()
            proc.comp_threshold_db = self._comp_thresh_var.get()
            proc.comp_ratio = self._comp_ratio_var.get()
            proc.comp_makeup_db = self._comp_makeup_var.get()
            
    def _update_meters(self):
        """
        Update the input/output level meters.
        
        LEARNING NOTE:
        Level meters convert dB values to visual bar widths.
        We map the range -60dB to 0dB onto the canvas width.
        Colors indicate level:
        - Green:  safe levels (-60 to -12 dB)
        - Yellow: getting hot (-12 to -6 dB)
        - Red:    near clipping (-6 to 0 dB)
        """
        if not self.is_dictating or not self.engine or not self.engine.processor:
            self.root.after(100, self._update_meters)
            return
            
        proc = self.engine.processor
        
        def draw_meter(canvas, level_db):
            canvas.delete("all")
            width = canvas.winfo_width()
            # Map -60..0 dB to 0..width pixels
            normalized = max(0.0, min(1.0, (level_db + 60) / 60.0))
            bar_width = int(normalized * width)
            
            if bar_width > 0:
                # Color based on level
                if normalized < 0.7:       # Below -12dB
                    color = self.colors["green"]
                elif normalized < 0.85:    # -12 to -6dB
                    color = self.colors["yellow"]
                else:                       # Above -6dB
                    color = self.colors["red"]
                canvas.create_rectangle(0, 0, bar_width, 12, fill=color, outline="")
        
        draw_meter(self.input_meter, proc.input_level_db if hasattr(proc, 'input_level_db') else -100)
        draw_meter(self.output_meter, proc.output_level_db if hasattr(proc, 'output_level_db') else -100)
        
        # Update gate status indicator
        if proc.gate_enabled:
            if proc.gate_is_open:
                self.gate_status_label.config(text="Gate: OPEN", fg=self.colors["green"])
            else:
                self.gate_status_label.config(text="Gate: CLOSED", fg=self.colors["red"])
        else:
            self.gate_status_label.config(text="Gate: OFF", fg=self.colors["subtext"])
        
        self.root.after(50, self._update_meters)

    def _build_text_area(self):
        """Build the main text editing area with a toggleable commands panel."""
        # Container that holds both the text area and the commands panel
        self.content_frame = tk.Frame(self.root, bg=self.colors["bg"])
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        
        # Main text area (left side)
        self.text_area = scrolledtext.ScrolledText(
            self.content_frame,
            wrap=tk.WORD,
            font=("Consolas", 13),
            bg=self.colors["bg"],
            fg=self.colors["text"],
            insertbackground=self.colors["accent"],
            selectbackground=self.colors["blue"],
            selectforeground=self.colors["bg"],
            padx=15,
            pady=10,
            undo=True,
            borderwidth=0,
            highlightthickness=0,
        )
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Commands reference panel (right side, starts hidden)
        self._commands_panel_visible = False
        self._build_commands_panel()
    
    def _build_commands_panel(self):
        """
        Build the voice commands reference panel.
        
        LEARNING NOTE:
        This is a scrollable panel that shows all available voice commands
        grouped by category. It sits to the right of the text area and
        can be toggled on/off without disrupting your dictation flow.
        
        Using a Canvas with a Frame inside is the standard Tkinter pattern
        for making a scrollable panel of widgets (since Frame itself 
        doesn't support scrolling).
        """
        self.commands_panel = tk.Frame(self.content_frame, bg="#252538", width=260)
        # Don't pack yet — starts hidden
        
        # Header
        header = tk.Frame(self.commands_panel, bg="#252538")
        header.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        tk.Label(header, text="Voice Commands",
                 font=("Segoe UI", 12, "bold"),
                 bg="#252538", fg=self.colors["text"]).pack(side=tk.LEFT)
        
        tk.Button(header, text="✕", font=("Segoe UI", 10),
                  bg="#252538", fg=self.colors["subtext"],
                  activebackground="#252538", activeforeground=self.colors["red"],
                  bd=0, cursor="hand2",
                  command=self._toggle_commands_panel).pack(side=tk.RIGHT)
        
        # Scrollable area
        canvas = tk.Canvas(self.commands_panel, bg="#252538",
                          highlightthickness=0, width=244)
        scrollbar = ttk.Scrollbar(self.commands_panel, orient=tk.VERTICAL,
                                   command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#252538")
        
        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scroll_frame.bind("<MouseWheel>", _on_mousewheel)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Populate commands by category
        categories = [
            ("Punctuation", [
                ("period / full stop", "."),
                ("comma", ","),
                ("question mark", "?"),
                ("exclamation mark", "!"),
                ("colon", ":"),
                ("semicolon", ";"),
                ("dash", "—"),
                ("hyphen", "-"),
                ("ellipsis", "..."),
            ]),
            ("Quotes & Brackets", [
                ("open quote", '"'),
                ("close quote", '"'),
                ("single quote", "'"),
                ("open parenthesis", "("),
                ("close parenthesis", ")"),
                ("open bracket", "["),
                ("close bracket", "]"),
            ]),
            ("Formatting", [
                ("new line", "↵ (line break)"),
                ("new paragraph", "↵↵ (double break)"),
                ("tab", "→ (tab)"),
            ]),
            ("Symbols", [
                ("at sign", "@"),
                ("hashtag", "#"),
                ("dollar sign", "$"),
                ("percent", "%"),
                ("ampersand", "&"),
                ("asterisk", "*"),
                ("slash", "/"),
                ("backslash", "\\"),
                ("underscore", "_"),
                ("equals", "="),
                ("plus sign", "+"),
            ]),
            ("Keyboard Shortcuts", [
                ("Ctrl+Shift+D", "Toggle dictation"),
                ("Ctrl+S", "Save file"),
                ("Ctrl+O", "Open file"),
                ("Ctrl+N", "New file"),
                ("F1", "Toggle this panel"),
            ]),
        ]
        
        for cat_name, commands in categories:
            # Category header
            cat_frame = tk.Frame(scroll_frame, bg="#252538")
            cat_frame.pack(fill=tk.X, padx=8, pady=(10, 2))
            
            tk.Label(cat_frame, text=cat_name.upper(),
                     font=("Segoe UI", 9, "bold"),
                     bg="#252538", fg=self.colors["accent"]).pack(anchor=tk.W)
            
            # Separator line
            tk.Frame(cat_frame, bg=self.colors["accent"],
                    height=1).pack(fill=tk.X, pady=(2, 0))
            
            # Command entries
            for say_text, result_text in commands:
                row = tk.Frame(scroll_frame, bg="#252538")
                row.pack(fill=tk.X, padx=12, pady=1)
                
                # "Say this" on the left
                tk.Label(row, text=f'"{say_text}"',
                         font=("Consolas", 9),
                         bg="#252538", fg=self.colors["green"],
                         anchor=tk.W).pack(side=tk.LEFT)
                
                # "Get this" on the right
                tk.Label(row, text=result_text,
                         font=("Consolas", 9, "bold"),
                         bg="#252538", fg=self.colors["yellow"],
                         anchor=tk.E).pack(side=tk.RIGHT)
        
        # Tip at the bottom
        tip_frame = tk.Frame(scroll_frame, bg="#252538")
        tip_frame.pack(fill=tk.X, padx=8, pady=(15, 10))
        tk.Label(tip_frame, 
                 text="💡 Commands are spoken as\nregular words while dictating.",
                 font=("Segoe UI", 9, "italic"),
                 bg="#252538", fg=self.colors["subtext"],
                 justify=tk.LEFT).pack(anchor=tk.W)
    
    def _toggle_commands_panel(self, event=None):
        """Show/hide the voice commands reference panel."""
        if self._commands_panel_visible:
            self.commands_panel.pack_forget()
            self.help_btn.config(text="❓ Commands")
            self._commands_panel_visible = False
        else:
            self.commands_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(3, 0))
            self.help_btn.config(text="✕ Commands")
            self._commands_panel_visible = True
        
    def _build_status_bar(self):
        """
        Build the bottom status bar with resource monitor and mic selector.
        
        Layout (two rows):
        ┌─────────────────────────────────────────────────────────┐
        │ [status message]                     [model status]     │
        │ CPU: 3.1%  RAM: 173.8 MB             🎤 Microphone Name │
        └─────────────────────────────────────────────────────────┘
        """
        status_outer = tk.Frame(self.root, bg=self.colors["surface"])
        status_outer.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Top row: status message + model info
        top_row = tk.Frame(status_outer, bg=self.colors["surface"])
        top_row.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        self.status_label = tk.Label(top_row,
                                      text="Ready — Press Ctrl+Shift+D to start dictation",
                                      font=("Consolas", 10),
                                      bg=self.colors["surface"],
                                      fg=self.colors["subtext"],
                                      anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)
        
        self.model_status = tk.Label(top_row,
                                      text="Model: Not loaded",
                                      font=("Consolas", 10),
                                      bg=self.colors["surface"],
                                      fg=self.colors["subtext"],
                                      anchor=tk.E)
        self.model_status.pack(side=tk.RIGHT)
        
        # Bottom row: CPU/RAM on left, mic input on right
        bottom_row = tk.Frame(status_outer, bg=self.colors["surface"])
        bottom_row.pack(fill=tk.X, padx=10, pady=(2, 5))
        
        # Resource monitor (bottom left)
        if HAS_PSUTIL:
            self._process = psutil.Process(os.getpid())
            
            self.cpu_label = tk.Label(bottom_row, text="CPU: --",
                                       font=("Consolas", 9),
                                       bg=self.colors["surface"],
                                       fg=self.colors["green"])
            self.cpu_label.pack(side=tk.LEFT, padx=(0, 10))
            
            self.ram_label = tk.Label(bottom_row, text="RAM: --",
                                       font=("Consolas", 9),
                                       bg=self.colors["surface"],
                                       fg=self.colors["green"])
            self.ram_label.pack(side=tk.LEFT, padx=(0, 10))
            
            # Start resource monitor polling
            self._update_resource_monitor()
        
        # Mic input selector (bottom right)
        self.mic_label = tk.Label(bottom_row, text="🎤",
                                   font=("Segoe UI", 9),
                                   bg=self.colors["surface"],
                                   fg=self.colors["subtext"])
        self.mic_label.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Get available input devices and populate dropdown
        self._mic_devices = self._get_input_devices()
        device_names = [d["name"] for d in self._mic_devices]
        
        if device_names:
            self.mic_var = tk.StringVar(value=device_names[0])
        else:
            device_names = ["No microphone found"]
            self.mic_var = tk.StringVar(value=device_names[0])
        
        self.mic_dropdown = ttk.Combobox(
            bottom_row,
            textvariable=self.mic_var,
            values=device_names,
            state="readonly",
            width=40,
            font=("Consolas", 9),
        )
        self.mic_dropdown.pack(side=tk.RIGHT, padx=(0, 2))
        self.mic_dropdown.bind("<<ComboboxSelected>>", self._on_mic_changed)
    
    def _get_input_devices(self):
        """
        Get a list of available audio input devices.
        
        LEARNING NOTE:
        PyAudio wraps the PortAudio library which talks to your OS's
        audio subsystem. Each device has properties like:
        - name: Human-readable device name
        - index: Numeric ID used when opening a stream
        - maxInputChannels: How many input channels (0 = output only)
        - defaultSampleRate: The device's preferred sample rate
        
        We filter for devices with maxInputChannels > 0 (input capable).
        """
        devices = []
        try:
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    devices.append({
                        "name": info["name"],
                        "index": i,
                    })
            pa.terminate()
        except Exception:
            pass
        return devices
    
    def _on_mic_changed(self, event=None):
        """Handle microphone selection change."""
        selected_name = self.mic_var.get()
        for dev in self._mic_devices:
            if dev["name"] == selected_name:
                self._selected_mic_index = dev["index"]
                # Update the engine's device index too
                if self.engine:
                    self.engine.device_index = self._selected_mic_index
                # If currently dictating, restart with new mic
                if self.is_dictating:
                    self.stop_dictation()
                    self.start_dictation()
                break
        
    def _build_partial_display(self):
        """
        Build the partial recognition display.
        
        LEARNING NOTE:
        This shows what Vosk is "thinking" in real-time as you speak.
        It updates constantly and gets replaced by the final result.
        This gives the user immediate visual feedback that the mic is working.
        """
        partial_frame = tk.Frame(self.root, bg="#2a2a3c", padx=10, pady=3)
        partial_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Label(partial_frame, text="HEARING:",
                 font=("Consolas", 9, "bold"),
                 bg="#2a2a3c", fg=self.colors["subtext"],
                 ).pack(side=tk.LEFT, padx=(5, 8))
        
        self.partial_label = tk.Label(
            partial_frame,
            text="(waiting for speech...)",
            font=("Consolas", 12, "italic"),
            bg="#2a2a3c",
            fg=self.colors["yellow"],
            anchor=tk.W,
            padx=5,
            pady=4,
        )
        self.partial_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
    def _browse_model(self):
        """Open a dialog to select the Vosk model directory."""
        path = filedialog.askdirectory(title="Select Vosk Model Directory")
        if path:
            self.model_var.set(path)
            
    def _load_model_async(self):
        """
        Load the Vosk model on a background thread with a progress indicator.
        
        LEARNING NOTE:
        Vosk's Model() constructor is a single blocking call — it doesn't
        give us incremental progress callbacks. So we use two strategies:
        
        1. Calculate the model folder size upfront so we can show it
        2. Monitor RAM growth during loading as a proxy for progress
           (the model gets loaded into memory, so RAM usage ≈ bytes loaded)
        3. Show elapsed time so the user knows it's not frozen
        
        For the large model (~1.8GB), loading can take 10-30 seconds.
        Without a progress indicator, the user thinks it crashed.
        """
        model_path = self.model_var.get().strip()
        if not model_path:
            messagebox.showerror("Error", "Please specify a model path.")
            return
            
        self.model_path = model_path
        
        # Calculate model size for progress estimation
        model_size_bytes = self._get_folder_size(self.model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Create progress bar overlay
        self._show_loading_overlay(model_size_mb)
        
        self.root.update()
        
        def load():
            try:
                start_time = time.time()
                
                # Get initial RAM usage for progress tracking
                if HAS_PSUTIL:
                    initial_ram = self._process.memory_info().rss
                
                # Update status: step 1 - loading model files
                self.root.after(0, lambda: self._update_loading_progress(
                    "Loading acoustic model...", 0, model_size_mb, 0))
                
                # Start a monitoring thread that tracks RAM growth
                loading_done = threading.Event()
                
                def monitor_progress():
                    """Track RAM growth as a proxy for model loading progress."""
                    while not loading_done.is_set():
                        elapsed = time.time() - start_time
                        if HAS_PSUTIL:
                            try:
                                current_ram = self._process.memory_info().rss
                                loaded_bytes = current_ram - initial_ram
                                loaded_mb = max(0, loaded_bytes / (1024 * 1024))
                                pct = min(95, (loaded_mb / model_size_mb * 100)) if model_size_mb > 0 else 0
                                self.root.after(0, lambda l=loaded_mb, p=pct, e=elapsed: 
                                    self._update_loading_progress(
                                        "Loading model into memory...", p, model_size_mb, e))
                            except Exception:
                                pass
                        else:
                            self.root.after(0, lambda e=elapsed:
                                self._update_loading_progress(
                                    "Loading model...", -1, model_size_mb, e))
                        loading_done.wait(timeout=0.3)
                
                monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
                monitor_thread.start()
                
                # Actually load the model (this is the blocking call)
                engine = AudioEngine(
                    model_path=self.model_path,
                    on_partial=self._on_partial,
                    on_final=self._on_final,
                    on_error=self._on_error,
                    device_index=self._selected_mic_index,
                )
                
                self.root.after(0, lambda: self._update_loading_progress(
                    "Loading acoustic model files...", 40, model_size_mb, 
                    time.time() - start_time))
                
                engine.load_model()
                
                # Signal monitor thread to stop
                loading_done.set()
                monitor_thread.join(timeout=1)
                
                elapsed = time.time() - start_time
                self.engine = engine
                self.root.after(0, lambda e=elapsed: self._on_model_loaded(True, elapsed=e))
                
            except Exception as e:
                loading_done.set()
                self.root.after(0, lambda: self._on_model_loaded(False, str(e)))
                
        threading.Thread(target=load, daemon=True).start()
    
    def _get_folder_size(self, path):
        """Get total size of a folder in bytes."""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
        except Exception:
            pass
        return total
    
    def _show_loading_overlay(self, model_size_mb):
        """
        Show a loading progress overlay in the text area.
        
        LEARNING NOTE:
        We overlay the progress bar on top of the text area rather than
        using a modal dialog, because modal dialogs block the event loop
        and we need the event loop running to update progress.
        """
        self.loading_frame = tk.Frame(self.root, bg=self.colors["bg"])
        self.loading_frame.place(relx=0.5, rely=0.45, anchor=tk.CENTER)
        
        # Title
        tk.Label(self.loading_frame, text="Loading Language Model",
                 font=("Segoe UI", 16, "bold"),
                 bg=self.colors["bg"], fg=self.colors["text"]).pack(pady=(0, 5))
        
        # Model size info
        tk.Label(self.loading_frame, text=f"Model size: {model_size_mb:.0f} MB",
                 font=("Consolas", 11),
                 bg=self.colors["bg"], fg=self.colors["subtext"]).pack(pady=(0, 10))
        
        # Progress bar
        self.style.configure("Loading.Horizontal.TProgressbar",
                             troughcolor=self.colors["surface"],
                             background=self.colors["green"],
                             thickness=25)
        
        self.loading_progress = ttk.Progressbar(
            self.loading_frame, 
            style="Loading.Horizontal.TProgressbar",
            length=400, 
            mode="determinate",
            maximum=100,
        )
        self.loading_progress.pack(pady=(0, 8))
        
        # Percentage + status text
        self.loading_pct_label = tk.Label(self.loading_frame, text="0%",
                                           font=("Consolas", 12, "bold"),
                                           bg=self.colors["bg"], 
                                           fg=self.colors["green"])
        self.loading_pct_label.pack(pady=(0, 3))
        
        self.loading_status_label = tk.Label(self.loading_frame, 
                                              text="Preparing...",
                                              font=("Consolas", 10),
                                              bg=self.colors["bg"],
                                              fg=self.colors["subtext"])
        self.loading_status_label.pack(pady=(0, 3))
        
        # Elapsed time
        self.loading_time_label = tk.Label(self.loading_frame, text="",
                                            font=("Consolas", 9),
                                            bg=self.colors["bg"],
                                            fg=self.colors["subtext"])
        self.loading_time_label.pack()
        
        # Disable dictation button while loading
        self.dictate_btn.config(state=tk.DISABLED)
        self.model_status.config(text="Model: Loading...")
        self.status_label.config(text="Loading model — please wait...")
    
    def _update_loading_progress(self, status_text, percentage, total_mb, elapsed):
        """Update the loading overlay with current progress."""
        if not hasattr(self, 'loading_frame') or not self.loading_frame.winfo_exists():
            return
        
        if percentage >= 0:
            self.loading_progress.config(mode="determinate")
            self.loading_progress["value"] = percentage
            self.loading_pct_label.config(text=f"{percentage:.0f}%")
        else:
            # No psutil — use indeterminate (pulsing) mode
            self.loading_progress.config(mode="indeterminate")
            self.loading_progress.step(5)
            self.loading_pct_label.config(text="...")
        
        self.loading_status_label.config(text=status_text)
        
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        if mins > 0:
            self.loading_time_label.config(text=f"Elapsed: {mins}m {secs}s")
        else:
            self.loading_time_label.config(text=f"Elapsed: {secs}s")
    
    def _hide_loading_overlay(self):
        """Remove the loading progress overlay."""
        if hasattr(self, 'loading_frame') and self.loading_frame.winfo_exists():
            self.loading_frame.destroy()
        self.dictate_btn.config(state=tk.NORMAL)
        
    def _on_model_loaded(self, success, error=None, elapsed=0):
        """Called when model loading finishes (on GUI thread)."""
        self._hide_loading_overlay()
        
        if success:
            model_name = os.path.basename(self.model_path)
            self.model_status.config(text=f"Model: {model_name} ✓")
            self.status_label.config(
                text=f"Model loaded in {elapsed:.1f}s! Press Ctrl+Shift+D or click 🎤 to dictate")
        else:
            self.model_status.config(text="Model: Failed ✗")
            self.status_label.config(text="Model loading failed — check path")
            messagebox.showerror("Model Error", f"Failed to load model:\n{error}")
            
    # ---- Dictation Control ----
    
    def toggle_dictation(self):
        """Start or stop dictation."""
        if self.is_dictating:
            self.stop_dictation()
        else:
            self.start_dictation()
            
    def start_dictation(self):
        """Begin capturing audio and recognizing speech."""
        if self.engine is None:
            # Try to auto-load model
            self._load_model_async()
            # We'll need the user to click again after model loads
            self.status_label.config(text="Loading model first — click Start again after it loads")
            return
            
        if self.is_dictating:
            return
            
        self.is_dictating = True
        self.processor.reset()
        self.engine.start()
        
        # Sync audio processing params from GUI to engine
        self._update_gate_params()
        self._update_comp_params()
        
        # Start level meter updates
        self._update_meters()
        
        self.dictate_btn.config(text="⏹ Stop Dictation")
        self.status_label.config(text="🔴 DICTATING — Speak now... (Ctrl+Shift+D to stop)")
        
    def stop_dictation(self):
        """Stop capturing audio."""
        if not self.is_dictating:
            return
            
        self.is_dictating = False
        
        if self.engine:
            # Stop the audio stream first so no new data comes in
            self.engine.stop()
            # Now safely flush any remaining text from the recognizer buffer
            try:
                self.engine.get_final_result()
            except Exception:
                pass  # Recognizer may already be cleared — that's fine
            
        self.dictate_btn.config(text="🎤 Start Dictation")
        self.status_label.config(text="Dictation stopped — Press Ctrl+Shift+D to resume")
        self.partial_label.config(text="")
        
    # ---- Callbacks from AudioEngine (called on audio thread!) ----
    # LEARNING NOTE:
    # These callbacks run on the AUDIO thread, NOT the GUI thread.
    # Tkinter is NOT thread-safe — you can't update widgets from other threads.
    # So we put the text into a thread-safe Queue, and the GUI thread polls it.
    
    def _on_partial(self, text):
        """Called when Vosk has a partial (in-progress) result."""
        self.partial_queue.put(text)
        
    def _on_final(self, text):
        """Called when Vosk has a final (confirmed) result."""
        processed = self.processor.process(text)
        self.text_queue.put(processed)
        
    def _on_error(self, error):
        """Called when an audio error occurs."""
        self.root.after(0, lambda: self._show_error(error))
        
    def _show_error(self, error):
        """Display error and stop dictation."""
        self.stop_dictation()
        messagebox.showerror("Audio Error", f"Microphone error:\n{error}")
        
    # ---- Queue Polling (GUI thread) ----
    
    def _poll_queues(self):
        """
        Check for new text from the audio thread every 50ms.
        
        LEARNING NOTE:
        This is the bridge between threads. Tkinter's `after()` method
        schedules a function to run on the GUI thread after a delay.
        We use this to safely update the text area with recognized speech.
        
        50ms (20 times/second) gives smooth updates without wasting CPU.
        """
        # Process all pending final results
        try:
            while True:
                text = self.text_queue.get_nowait()
                self._insert_text(text)
                self.partial_label.config(text="")  # Clear partial when final arrives
        except queue.Empty:
            pass
            
        # Update partial display (only show the latest)
        latest_partial = None
        try:
            while True:
                latest_partial = self.partial_queue.get_nowait()
        except queue.Empty:
            pass
        if latest_partial is not None:
            self.partial_label.config(text=latest_partial)
        elif not self.is_dictating and hasattr(self, 'partial_label'):
            # Show idle message when not dictating
            current = self.partial_label.cget("text")
            if current and current != "(waiting for speech...)":
                pass  # Keep last text visible briefly
            elif not current:
                self.partial_label.config(text="(waiting for speech...)")
            
        # Schedule next poll
        self.root.after(50, self._poll_queues)
        
        # Update level meters if dictating
        if self.is_dictating and self.engine and hasattr(self.engine, 'processor') and hasattr(self, 'input_level_label'):
            proc = self.engine.processor
            in_db = proc.input_level_db
            out_db = proc.output_level_db
            
            # Format the level display with a simple bar
            def db_bar(db, width=12):
                """Create a text-based level meter."""
                # Map -60dB..0dB to 0..width
                level = max(0, min(width, int((db + 60) / 60 * width)))
                return "█" * level + "░" * (width - level)
            
            self.input_level_label.config(
                text=f"IN: {db_bar(in_db)} {in_db:5.1f}dB"
            )
            self.output_level_label.config(
                text=f"OUT: {db_bar(out_db)} {out_db:5.1f}dB"
            )
    
    def _update_resource_monitor(self):
        """
        Update CPU and RAM usage display every 2 seconds.
        
        LEARNING NOTE:
        psutil gives us process-level resource usage:
        - cpu_percent(): CPU usage as a percentage of one core
          (can exceed 100% on multi-core if using multiple cores)
        - memory_info().rss: "Resident Set Size" — the actual RAM
          this process is using right now (in bytes)
        
        We update every 2 seconds instead of every frame because:
        1. psutil calls have some overhead
        2. CPU/RAM don't change fast enough to need 20Hz updates
        3. It keeps the status bar readable (not flickering)
        """
        if not HAS_PSUTIL:
            return
            
        try:
            cpu = self._process.cpu_percent(interval=None)
            mem = self._process.memory_info().rss
            
            # Format memory in appropriate units
            if mem >= 1024 * 1024 * 1024:
                mem_str = f"{mem / (1024**3):.1f} GB"
            else:
                mem_str = f"{mem / (1024**2):.1f} MB"
            
            self.cpu_label.config(text=f"CPU: {cpu:4.1f}%")
            self.ram_label.config(text=f"RAM: {mem_str}")
        except Exception:
            pass
        
        # Schedule next update (every 2 seconds)
        self.root.after(2000, self._update_resource_monitor)
        
    def _insert_text(self, text):
        """Insert recognized text at the cursor position in the text area."""
        if not text:
            return
            
        # Get current cursor position
        cursor_pos = self.text_area.index(tk.INSERT)
        
        # Check if we need a space before the new text
        if cursor_pos != "1.0":  # Not at the very beginning
            char_before = self.text_area.get(f"{cursor_pos}-1c", cursor_pos)
            # Add space unless the previous char is a newline or opening bracket
            if char_before and char_before not in ("\n", "\t", "(", "[", '"', "'", " "):
                # And the new text doesn't start with punctuation
                if text[0] not in (".", ",", "!", "?", ":", ";", ")", "]", '"', "'", "\n"):
                    text = " " + text
                    
        self.text_area.insert(tk.INSERT, text)
        self.text_area.see(tk.INSERT)  # Auto-scroll to cursor
        
    # ---- File Operations ----
    
    def new_file(self):
        """Clear the text area for a new document."""
        if self.text_area.get("1.0", tk.END).strip():
            if not messagebox.askyesno("New File", "Discard current text?"):
                return
        self.text_area.delete("1.0", tk.END)
        self.processor.reset()
        
    def open_file(self):
        """Open a text file into the editor."""
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert("1.0", content)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file:\n{e}")
                
    def save_file(self):
        """Save the text area content to a file."""
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            try:
                content = self.text_area.get("1.0", tk.END)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                self.status_label.config(text=f"Saved: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")
                
    # ---- Global Hotkey ----
    
    def _setup_global_hotkey(self):
        """
        Set up Ctrl+Shift+D as a global hotkey to toggle dictation.
        
        LEARNING NOTE:
        pynput's keyboard listener runs on its own thread and can detect
        key presses system-wide, even when our window isn't focused.
        This is how Windows' Win+H works — it's a global hotkey.
        
        We use Ctrl+Shift+D instead of Win+H to avoid conflicting with
        the built-in Windows dictation feature.
        """
        self._pressed_keys = set()
        
        def on_press(key):
            self._pressed_keys.add(key)
            # Check for Ctrl+Shift+D
            ctrl = (pynput_keyboard.Key.ctrl_l in self._pressed_keys or
                    pynput_keyboard.Key.ctrl_r in self._pressed_keys)
            shift = (pynput_keyboard.Key.shift_l in self._pressed_keys or
                     pynput_keyboard.Key.shift_r in self._pressed_keys)
            d_key = False
            try:
                if hasattr(key, 'char') and key.char and key.char.lower() == 'd':
                    d_key = True
            except AttributeError:
                pass
                
            if ctrl and shift and d_key:
                # Use after() to run on GUI thread
                self.root.after(0, self.toggle_dictation)
                
        def on_release(key):
            self._pressed_keys.discard(key)
            
        self._hotkey_listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()
        
    # ---- Cleanup ----
    
    def _on_close(self):
        """Clean up resources when the window is closed."""
        self.stop_dictation()
        if hasattr(self, '_hotkey_listener'):
            self._hotkey_listener.stop()
        self.root.destroy()
        
    def run(self):
        """Start the application."""
        self.root.mainloop()


# ===========================================================================
# MODEL DOWNLOADER
# ===========================================================================
# LEARNING NOTE:
# Vosk models need to be downloaded separately. This helper function
# downloads and extracts the model automatically if it's not found.
# ===========================================================================

def download_model(model_name="vosk-model-small-en-us-0.15", target_dir="."):
    """
    Download a Vosk model if not already present.
    
    Available English models:
    - vosk-model-small-en-us-0.15  (~50MB)  — Fast, good for real-time
    - vosk-model-en-us-0.22        (~1.8GB) — High accuracy
    - vosk-model-en-us-0.42-gigaspeech (~2.3GB) — Best accuracy
    
    For other languages, see: https://alphacephei.com/vosk/models
    """
    import urllib.request
    import zipfile
    
    model_path = os.path.join(target_dir, model_name)
    
    if os.path.exists(model_path):
        print(f"Model already exists at: {model_path}")
        return model_path
        
    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    zip_path = os.path.join(target_dir, f"{model_name}.zip")
    
    print(f"Downloading {model_name}...")
    print(f"URL: {url}")
    print("This may take a few minutes depending on your connection...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  [{pct:3d}%] {mb_down:.1f} / {mb_total:.1f} MB", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
        print("\nDownload complete! Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        os.remove(zip_path)  # Clean up zip file
        print(f"Model ready at: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"Please download manually from: {url}")
        print(f"Extract to: {target_dir}")
        return None


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    """
    Main entry point. Handles model setup and launches the app.
    
    Usage:
        python dictator.py                    # Uses default model path
        python dictator.py --model PATH       # Specify model directory
        python dictator.py --download         # Download the small English model
        python dictator.py --download-large   # Download the large English model
        python dictator.py --download-giga    # Download the gigaspeech model (best)
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Basic Speech-to-Text with Tkinter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dictator.py --download              Download small English model (~50MB)
  python dictator.py --download-large        Download large English model (~1.8GB)
  python dictator.py --download-giga         Download gigaspeech model (~2.3GB, best)
  python dictator.py --model vosk-model-small-en-us-0.15
  python dictator.py                         Launch with default model path
  
Voice Commands while dictating:
  "period"            → .        "comma"              → ,
  "question mark"     → ?        "exclamation mark"   → !
  "new line"          → newline  "new paragraph"      → double newline
  "colon"             → :        "semicolon"          → ;
  "open parenthesis"  → (        "close parenthesis"  → )
  "open quote"        → "        "close quote"        → "
  
Keyboard Shortcuts:
  Ctrl+Shift+D    Toggle dictation (global hotkey)
  Ctrl+S          Save file
  Ctrl+O          Open file
  Ctrl+N          New file
        """
    )
    parser.add_argument("--model", default="vosk-model-small-en-us-0.15",
                        help="Path to Vosk model directory")
    parser.add_argument("--download", action="store_true",
                        help="Download the small English model (~50MB)")
    parser.add_argument("--download-large", action="store_true",
                        help="Download the large English model (~1.8GB)")
    parser.add_argument("--download-giga", action="store_true",
                        help="Download the gigaspeech model (~2.3GB, best accuracy)")
    
    args = parser.parse_args()
    
    # Handle model downloads
    if args.download:
        path = download_model("vosk-model-small-en-us-0.15")
        if path is None:
            sys.exit(1)
        args.model = path
        
    if args.download_large:
        path = download_model("vosk-model-en-us-0.22")
        if path is None:
            sys.exit(1)
        args.model = path
    
    if args.download_giga:
        path = download_model("vosk-model-en-us-0.42-gigaspeech")
        if path is None:
            sys.exit(1)
        args.model = path
    
    # Launch the app
    print("=" * 50)
    print("  Basic Speech-to-Text with Tkinter")
    print("=" * 50)
    print(f"  Model path: {args.model}")
    print(f"  Global hotkey: Ctrl+Shift+D")
    print("=" * 50)
    
    app = SpeechToTextApp()
    app.model_var.set(args.model)
    
    # Auto-load model if it exists
    if os.path.exists(args.model):
        app.root.after(500, app._load_model_async)  # Load after GUI is up
    else:
        print(f"\nModel not found at '{args.model}'")
        print("Run with --download to get the small English model")
        print("Run with --download-large for better accuracy")
        print("Run with --download-giga for best accuracy")
        print("Or click 'Browse' in the app to select your model directory")
    
    app.run()


if __name__ == "__main__":
    main()
