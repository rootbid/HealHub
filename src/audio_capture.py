import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import threading
import time
import wave
from scipy import signal
from scipy.io import wavfile
import requests
import io
import librosa

class AudioCleanerUsingLib:
    """
    Cleans audio for speech recognition:
    - Mono conversion
    - Resampling
    - Silence trimming
    - Noise reduction
    - Normalization
    """

    def __init__(self, target_sr=16000, target_dbfs=-20):
        self.target_sr = target_sr
        self.target_dbfs = target_dbfs

    def get_cleaned_audio(self, data, sr):
        # 1. Convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # 2. Resample
        if sr != self.target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # 3. Trim silence
        data, _ = librosa.effects.trim(data, top_db=25)

        # 4. Noise reduction (simple median filter)
        data = signal.medfilt(data, kernel_size=3)

        # 5. Normalize
        rms = np.sqrt(np.mean(data**2))
        if rms > 0:
            scalar = 10 ** (self.target_dbfs / 20) / rms
            data = data * scalar

        return data, sr

class AudioCleaner:
    """Audio processing utilities for cleaning speech audio"""
    
    @staticmethod
    def remove_silence(audio_data, sample_rate, silence_threshold=0.01, min_silence_duration=0.5):
        """Remove silence segments from audio"""
        # Calculate frame size for the minimum silence duration
        frame_size = int(min_silence_duration * sample_rate)
        
        # Calculate RMS energy for each frame
        frames = []
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]
            if len(frame) > 0:
                rms = np.sqrt(np.mean(frame**2))
                frames.append((i, i + len(frame), rms > silence_threshold))
        
        # Keep only non-silent frames
        cleaned_audio = []
        for start, end, is_voice in frames:
            if is_voice:
                cleaned_audio.append(audio_data[start:end])
        
        return np.concatenate(cleaned_audio) if cleaned_audio else np.array([])
    
    @staticmethod
    def apply_noise_reduction(audio_data, sample_rate):
        """Apply basic noise reduction using spectral subtraction"""
        # Apply a high-pass filter to remove low-frequency noise
        nyquist = sample_rate / 2
        high_cutoff = 80  # Remove frequencies below 80 Hz
        
        if high_cutoff < nyquist:
            sos = signal.butter(4, high_cutoff / nyquist, btype='high', output='sos')
            filtered_audio = signal.sosfilt(sos, audio_data)
        else:
            filtered_audio = audio_data
        
        return filtered_audio
    
    @staticmethod
    def normalize_audio(audio_data, target_level=0.7):
        """Normalize audio to target level"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            normalized = audio_data * (target_level / max_val)
            return normalized
        return audio_data
    
    @staticmethod
    def apply_voice_enhancement(audio_data, sample_rate):
        """Enhance voice frequencies (300-3400 Hz)"""
        nyquist = sample_rate / 2
        low_cutoff = 300 / nyquist
        high_cutoff = 3400 / nyquist
        
        # Band-pass filter for voice frequencies
        sos = signal.butter(4, [low_cutoff, high_cutoff], btype='band', output='sos')
        enhanced_audio = signal.sosfilt(sos, audio_data)
        
        return enhanced_audio

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
                    device=10  # Stereo Microphone
                )
                print(f"Using Stereo Microphone (device 10) at {self.sample_rate} Hz")
            
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

class SarvamSTTIntegration:
    """Integration class for Sarvam's Saarika v2 STT model"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.api_url = "https://api.sarvam.ai/speech-to-text" 
        # Note: Replace with actual Sarvam API endpoint and implementation
        
    def transcribe_audio(self, audio_data, sample_rate=48000, source_language="hi-IN"):
        """
        Send cleaned audio to Sarvam's Saarika v2 for transcription
        
        Args:
            audio_data: Cleaned audio data (int16)
            sample_rate: Audio sample rate
            source_language: Source language code (e.g., "hi-IN", "ta-IN", etc.)
        """
        
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
        audio_buffer.seek(0)

        headers = {
            "api-subscription-key": self.api_key
        }
        files = {
            "file": ("audio.wav", audio_buffer, "audio/wav")
        }
        data = {
            "language": source_language
        }

        try:
            response = requests.post(self.api_url, headers=headers, files=files, data=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            transcription = result.get("transcript", "")
            language_detected = result.get("language_code", source_language)
            return {
                "transcription": transcription,
                "language_detected": language_detected
            }
        except requests.RequestException as e:
            print(f"❌ Sarvam STT API call failed: {e}")
            return {
                "transcription": "",
                "language_detected": source_language
            }

# Main usage example
def main():
    """Example usage of the clean audio capture system"""
    
    # Initialize components
    audio_capture = CleanAudioCapture(sample_rate=48000)
    stt_service = SarvamSTTIntegration()
    
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
            result = stt_service.transcribe_audio(
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