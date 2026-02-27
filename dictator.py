"""
=============================================================================
VoicePad - A Real-Time Speech Recognition Dictation App
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
import wave
import struct

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
    
    def __init__(self, model_path, on_partial=None, on_final=None, on_error=None):
        """
        Args:
            model_path: Path to the Vosk language model directory
            on_partial: Callback function for partial (in-progress) results
            on_final: Callback function for final (confirmed) results  
            on_error: Callback function for error reporting
        """
        self.model_path = model_path
        self.on_partial = on_partial or (lambda text: None)
        self.on_final = on_final or (lambda text: None)
        self.on_error = on_error or (lambda err: None)
        
        self.model = None
        self.recognizer = None
        self.audio = None
        self.stream = None
        self.is_listening = False
        self._thread = None
        
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
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            self.stream.start_stream()
            
            while self.is_listening:
                try:
                    # Read audio data from microphone
                    # This blocks until CHUNK_SIZE frames are available
                    data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    
                    # Feed audio to Vosk recognizer
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
                    raise
                    
        except Exception as e:
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

class VoicePadApp:
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
        
        # ---- Build GUI ----
        self.root = tk.Tk()
        self.root.title("VoicePad — Speech Dictation")
        self.root.geometry("900x650")
        self.root.minsize(600, 400)
        
        # Set icon and styling
        self.root.configure(bg="#1e1e2e")
        self._setup_styles()
        self._build_toolbar()
        self._build_text_area()
        self._build_status_bar()
        self._build_partial_display()
        
        # ---- Keyboard Shortcuts ----
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-n>", lambda e: self.new_file())
        
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
        
    def _build_text_area(self):
        """Build the main text editing area."""
        text_frame = tk.Frame(self.root, bg=self.colors["bg"])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))
        
        self.text_area = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 13),
            bg=self.colors["bg"],
            fg=self.colors["text"],
            insertbackground=self.colors["accent"],  # cursor color
            selectbackground=self.colors["blue"],
            selectforeground=self.colors["bg"],
            padx=15,
            pady=10,
            undo=True,
            borderwidth=0,
            highlightthickness=0,
        )
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
    def _build_status_bar(self):
        """Build the bottom status bar."""
        status_frame = ttk.Frame(self.root, style="Status.TFrame", padding=(10, 5))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_frame,
                                       text="Ready — Press Ctrl+Shift+D to start dictation",
                                       style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT)
        
        self.model_status = ttk.Label(status_frame,
                                       text="Model: Not loaded",
                                       style="Status.TLabel")
        self.model_status.pack(side=tk.RIGHT)
        
    def _build_partial_display(self):
        """
        Build the partial recognition display.
        
        LEARNING NOTE:
        This shows what Vosk is "thinking" in real-time as you speak.
        It updates constantly and gets replaced by the final result.
        This gives the user immediate visual feedback that the mic is working.
        """
        partial_frame = tk.Frame(self.root, bg=self.colors["surface"])
        partial_frame.pack(fill=tk.X, side=tk.BOTTOM, before=self.root.winfo_children()[-1])
        
        self.partial_label = tk.Label(
            partial_frame,
            text="",
            font=("Consolas", 11, "italic"),
            bg=self.colors["surface"],
            fg=self.colors["yellow"],
            anchor=tk.W,
            padx=15,
            pady=5,
        )
        self.partial_label.pack(fill=tk.X)
        
    def _browse_model(self):
        """Open a dialog to select the Vosk model directory."""
        path = filedialog.askdirectory(title="Select Vosk Model Directory")
        if path:
            self.model_var.set(path)
            
    def _load_model_async(self):
        """Load the Vosk model on a background thread (it can take a few seconds)."""
        model_path = self.model_var.get().strip()
        if not model_path:
            messagebox.showerror("Error", "Please specify a model path.")
            return
            
        self.model_path = model_path
        self.model_status.config(text="Model: Loading...")
        self.status_label.config(text="Loading model — please wait...")
        self.root.update()
        
        def load():
            try:
                self.engine = AudioEngine(
                    model_path=self.model_path,
                    on_partial=self._on_partial,
                    on_final=self._on_final,
                    on_error=self._on_error,
                )
                self.engine.load_model()
                self.root.after(0, lambda: self._on_model_loaded(True))
            except Exception as e:
                self.root.after(0, lambda: self._on_model_loaded(False, str(e)))
                
        threading.Thread(target=load, daemon=True).start()
        
    def _on_model_loaded(self, success, error=None):
        """Called when model loading finishes (on GUI thread)."""
        if success:
            model_name = os.path.basename(self.model_path)
            self.model_status.config(text=f"Model: {model_name} ✓")
            self.status_label.config(text="Model loaded! Press Ctrl+Shift+D or click 🎤 to dictate")
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
        
        self.dictate_btn.config(text="⏹ Stop Dictation")
        self.status_label.config(text="🔴 DICTATING — Speak now... (Ctrl+Shift+D to stop)")
        
    def stop_dictation(self):
        """Stop capturing audio."""
        if not self.is_dictating:
            return
            
        self.is_dictating = False
        
        if self.engine:
            # Get any remaining buffered text
            self.engine.get_final_result()
            self.engine.stop()
            
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
            self.partial_label.config(text=f"  ⟩ {latest_partial}")
            
        # Schedule next poll
        self.root.after(50, self._poll_queues)
        
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
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VoicePad — Real-Time Speech Dictation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dictator.py --download              Download small English model (~50MB)
  python dictator.py --download-large        Download large English model (~1.8GB)
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
    
    # Launch the app
    print("=" * 50)
    print("  VoicePad — Speech Dictation App")
    print("=" * 50)
    print(f"  Model path: {args.model}")
    print(f"  Global hotkey: Ctrl+Shift+D")
    print("=" * 50)
    
    app = VoicePadApp()
    app.model_var.set(args.model)
    
    # Auto-load model if it exists
    if os.path.exists(args.model):
        app.root.after(500, app._load_model_async)  # Load after GUI is up
    else:
        print(f"\nModel not found at '{args.model}'")
        print("Run with --download to get the small English model")
        print("Or click 'Browse' in the app to select your model directory")
    
    app.run()


if __name__ == "__main__":
    main()
