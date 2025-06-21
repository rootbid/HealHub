import sounddevice as sd
import numpy as np
import queue
import threading
import time
import wave
from scipy import signal
from scipy.io import wavfile
import requests
import io

try:
    from src.utils import HealHubUtilities
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.utils import HealHubUtilities

class AudioCleaner:
    """
    Audio Cleaner:
    - Mono conversion
    - Resampling (using scipy)
    - Silence removal (using RMS)
    - Noise reduction (median filter + high-pass filter)
    - Voice frequency enhancement
    - Normalization (target dBFS or peak level)
    """

    def __init__(self, target_sr=16000, target_dbfs=-20, target_level=0.7):
        self.target_sr = target_sr
        self.target_dbfs = target_dbfs
        self.target_level = target_level

    @staticmethod
    def convert_to_mono(data):
        if len(data.shape) > 1:
            return np.mean(data, axis=1)
        return data

    @staticmethod
    def resample_audio(data, sr, target_sr):
        if sr != target_sr:
            gcd = np.gcd(sr, target_sr)
            up = target_sr // gcd
            down = sr // gcd
            data = signal.resample_poly(data, up, down)
        return data, target_sr

    @staticmethod
    def remove_silence_rms(data, sr, silence_threshold=0.01, min_silence_duration=0.2):
        frame_size = int(min_silence_duration * sr)
        frames = []
        for i in range(0, len(data), frame_size):
            frame = data[i:i + frame_size]
            if len(frame) > 0:
                rms = np.sqrt(np.mean(frame**2))
                frames.append((i, i + len(frame), rms > silence_threshold))

        cleaned_audio = [data[start:end] for start, end, is_voice in frames if is_voice]
        return np.concatenate(cleaned_audio) if cleaned_audio else np.array([])

    @staticmethod
    def apply_noise_reduction(data, sr, median_filter=True, high_pass=True):
        if median_filter:
            data = signal.medfilt(data, kernel_size=3)
        if high_pass:
            nyquist = sr / 2
            cutoff = 80  # Hz
            if cutoff < nyquist:
                sos = signal.butter(4, cutoff / nyquist, btype='high', output='sos')
                data = signal.sosfilt(sos, data)
        return data

    @staticmethod
    def enhance_voice_frequencies(data, sr, low_freq=300, high_freq=3400):
        nyquist = sr / 2
        sos = signal.butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band', output='sos')
        return signal.sosfilt(sos, data)

    @staticmethod
    def normalize_audio_dbfs(data, target_dbfs):
        rms = np.sqrt(np.mean(data**2))
        if rms > 0:
            scalar = 10 ** (target_dbfs / 20) / rms
            data = data * scalar
        return data

    @staticmethod
    def normalize_audio_peak(data, target_level):
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data * (target_level / max_val)
        return data

    def get_cleaned_audio(self, data, sr, use_rms_silence_removal=True, apply_voice_enhance=True):
        # Step 1: Mono conversion
        data = self.convert_to_mono(data)

        # Step 2: Resample
        data, sr = self.resample_audio(data, sr, self.target_sr)

        # Step 3: Silence removal
        if use_rms_silence_removal:
            data = self.remove_silence_rms(data, sr)

        # Step 4: Noise reduction
        data = self.apply_noise_reduction(data, sr)

        # Step 5: Voice enhancement (optional)
        if apply_voice_enhance:
            data = self.enhance_voice_frequencies(data, sr)

        # Step 6: Normalization (DBFS and peak)
        data = self.normalize_audio_dbfs(data, self.target_dbfs)
        data = self.normalize_audio_peak(data, self.target_level)

        return data, sr

class CleanAudioCapture:
    def __init__(self, sample_rate=48000, channels=1, dtype=np.int16):
        print("Initializing CleanAudioCapture with parameters:")
        print(f"- Sample rate: {sample_rate}")
        print(f"- Channels: {channels}")
        print(f"- Data type: {dtype}")
        
        # Query supported sample rates for devices
        try:
            devices = sd.query_devices()
            print("\nAvailable audio devices and their supported sample rates:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Only show input devices
                    try:
                        supported_rates = sd.query_devices(device=i)['default_samplerate']
                        print(f"{i}: {device['name']} (in={device['max_input_channels']}, out={device['max_output_channels']})")
                        print(f"   Supported sample rate: {supported_rates} Hz")
                    except Exception as e:
                        print(f"{i}: {device['name']} - Error querying sample rate: {e}")
        except Exception as e:
            print(f"Error querying audio devices: {e}")
        
        # Use a standard sample rate that's widely supported
        self.sample_rate = 48000  # Most devices support 44.1kHz
        self.channels = channels
        self.dtype = dtype
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.cleaner = AudioCleaner()
        
        # Voice activity detection parameters - adjusted for better sensitivity
        self.voice_threshold = 0.01  # Lowered threshold for better voice detection
        self.silence_duration = 3.0  # Increased silence duration
        self.last_voice_time = time.time()
        self.voice_detected = False
        self.total_frames_processed = 0
        self.voice_frames_detected = 0

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input with voice activity detection"""
        if status:
            print(f"Audio input error: {status}")
        
        # Calculate volume (RMS) for voice activity detection
        volume = np.sqrt(np.mean(indata**2))
        self.total_frames_processed += 1
        
        # Log volume levels periodically
        if self.total_frames_processed % 100 == 0:
            print(f"Current volume level: {volume:.6f} (threshold: {self.voice_threshold})")
        
        # Voice activity detection
        if volume > self.voice_threshold:
            self.voice_detected = True
            self.voice_frames_detected += 1
            self.last_voice_time = time.inputBufferAdcTime
            if self.voice_frames_detected % 100 == 0:
                print(f"Voice detected! Volume: {volume:.6f}")
        
        # Add audio to queue if voice is detected
        if self.voice_detected:
            audio_data = (indata * 32767).astype(self.dtype)
            self.audio_queue.put(audio_data.copy())
            
        # Check for silence timeout
        current_time = time.inputBufferAdcTime
        if (current_time - self.last_voice_time) > self.silence_duration:
            if self.voice_detected:
                print(f"Silence detected after {self.voice_frames_detected} voice frames. Processing audio...")
                self.stop_recording()
    
    def start_recording(self):
        """Start real-time audio capture with voice activity detection"""
        print("\nStarting audio recording...")
        self.is_recording = True
        self.voice_detected = False
        self.last_voice_time = time.time()
        self.total_frames_processed = 0
        self.voice_frames_detected = 0
        
        try:
            print("Creating audio input stream...")
            # Try to use the digital microphone first, fall back to stereo microphone if needed
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32,
                    callback=self.audio_callback,
                    blocksize=1024,
                )
            except Exception as mic_error:
                print(f"Failed to use Digital Microphone: {mic_error}")
                print("Trying Stereo Microphone...")
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=np.float32,
                    callback=self.audio_callback,
                    blocksize=1024,
                )
            
            print("Starting audio stream...")
            self.stream.start()
            print("Audio stream started successfully")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self):
        """Stop audio capture"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        print("🛑 Recording stopped.")
        
    def get_raw_audio_buffer(self):
        """Get raw audio data from queue"""
        audio_chunks = []
        while not self.audio_queue.empty():
            audio_chunks.append(self.audio_queue.get())
        
        if audio_chunks:
            return np.concatenate(audio_chunks, axis=0)
        return np.array([])
    
    def get_cleaned_audio(self, apply_enhancement=True):
        """Get cleaned and processed audio ready for STT"""
        raw_audio = self.get_raw_audio_buffer()
        
        if len(raw_audio) == 0:
            print("No audio data in buffer")
            return np.array([])
            
        print(f"Processing {len(raw_audio)} samples of audio...")
        
        # Convert to float for processing
        audio_float = raw_audio.astype(np.float32) / 32767.0
        
        # Resample to 16kHz if needed for STT
        if self.sample_rate != 16000:
            print(f"Resampling from {self.sample_rate} Hz to 16000 Hz...")
            audio_float = signal.resample(audio_float, round(len(audio_float) * 16000 / self.sample_rate))
        
        # Step 1: Remove silence segments
        cleaned_audio = self.cleaner.remove_silence(
            audio_float, 
            16000,  # Use 16kHz for STT
            silence_threshold=0.01,  # Lowered threshold
            min_silence_duration=0.3
        )
        
        if len(cleaned_audio) == 0:
            print("⚠️ No voice detected in audio after cleaning")
            return np.array([])
        
        print(f"Audio cleaned successfully: {len(raw_audio)} → {len(cleaned_audio)} samples")
        return cleaned_audio
    
    def save_audio(self, audio_data, filename):
        """Save audio data to WAV file"""
        if len(audio_data) > 0:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            print(f"💾 Audio saved to {filename}")
        else:
            print("⚠️ No audio data to save")

# Main usage example
def main():
    """Example usage of the clean audio capture system"""
    
    # Initialize components
    audio_capture = CleanAudioCapture(sample_rate=48000)
    util = HealHubUtilities()
    
    try:
        # Start recording
        audio_capture.start_recording()
        
        # Wait for recording to complete (voice activity detection will stop it)
        while audio_capture.is_recording:
            time.sleep(0.1)
        
        # Get cleaned audio
        cleaned_audio = audio_capture.get_cleaned_audio(apply_enhancement=True)
        
        if len(cleaned_audio) > 0:
            # Save for debugging
            audio_capture.save_audio(cleaned_audio, "cleaned_audio.wav")
            
            # Send to Sarvam STT
            result = util.transcribe_audio(
                cleaned_audio, 
                sample_rate=audio_capture.sample_rate,
                source_language="hi-IN"  # Change as needed
            )
            
            print(f"🎯 Transcription: {result['transcription']}")
            
        else:
            print("❌ No valid audio captured")
            
    except KeyboardInterrupt:
        audio_capture.stop_recording()
        print("\n👋 Recording interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()