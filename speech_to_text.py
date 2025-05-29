import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import tempfile
import os
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load Whisper model
model = whisper.load_model("base")

# Supported languages
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Bengali": "bn", "Tamil": "ta",
    "Telugu": "te", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa",
    "Malayalam": "ml", "Kannada": "kn", "Urdu": "ur"
}

SAMPLE_RATE = 16000
recording = False
audio_frames = []

def start_recording():
    global recording, audio_frames
    audio_frames = []
    recording = True

    def callback(indata, frames, time, status):
        if recording:
            audio_frames.append(indata.copy())
            update_waveform(indata)

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
    stream.start()
    record_button.stream = stream

def stop_recording():
    global recording
    recording = False
    stream = getattr(record_button, "stream", None)
    if stream:
        stream.stop()
        stream.close()

    audio_data = np.concatenate(audio_frames, axis=0)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    write(path, SAMPLE_RATE, audio_data)
    threading.Thread(target=transcribe_audio, args=(path,)).start()

def transcribe_audio(audio_path):
    lang_name = language_var.get()
    output_mode = output_var.get()
    language_code = LANGUAGES.get(lang_name, "en")
    result_text.set("Transcribing...")

    try:
        result = model.transcribe(audio_path,
                                  language=language_code,
                                  task="translate" if output_mode == "Translate to English" else "transcribe")
        result_text.set(result["text"])
    except Exception as e:
        result_text.set("Transcription failed.")
        messagebox.showerror("Error", f"Transcription failed: {e}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# ---------- GUI Setup ----------

root = tk.Tk()
root.title("Voice-to-Text with Whisper")
root.geometry("500x600")

# Language selection
language_var = tk.StringVar(value="Hindi")
ttk.Label(root, text="Select Language:").pack(pady=5)
ttk.Combobox(root, textvariable=language_var, values=list(LANGUAGES.keys()), state="readonly").pack()

# Output mode: transcribe or translate
output_var = tk.StringVar(value="Transcribe in native script")
ttk.Label(root, text="Output Mode:").pack(pady=5)
ttk.Combobox(root, textvariable=output_var, values=["Transcribe in native script", "Translate to English"], state="readonly").pack()

# Waveform setup
fig, ax = plt.subplots(figsize=(4, 1.5))
line, = ax.plot([], [], lw=2)
ax.set_ylim(-1, 1)
ax.set_xlim(0, 400)
ax.axis("off")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(pady=15)

# Update waveform
def update_waveform(indata):
    data = indata.flatten()
    line.set_ydata(np.pad(data, (0, max(0, 400 - len(data))), 'constant'))
    line.set_xdata(np.arange(len(line.get_ydata())))
    canvas.draw()

# Record button
record_button = tk.Button(root, text="🎤 Hold to Record", width=25, bg="red", fg="white")
record_button.pack(pady=15)
record_button.bind('<ButtonPress>', lambda event: start_recording())
record_button.bind('<ButtonRelease>', lambda event: stop_recording())

# Result display
result_text = tk.StringVar()
tk.Label(root, text="Transcription:", font=("Arial", 12, "bold")).pack()
tk.Label(root, textvariable=result_text, wraplength=450, justify="left").pack(pady=10)

root.mainloop()
