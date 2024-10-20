import os
import time
import sounddevice as sd
import numpy as np
import threading
import logging
import wave
import io
import webrtcvad
from groq import Groq
from pydub import AudioSegment
from io import BytesIO
import queue
import collections
import tkinter as tk
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Groq client securely
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")
client = Groq(api_key=groq_api_key)

# Parameters
sample_rate = 16000  # 16 kHz
channels = 1  # Mono audio
frame_duration_ms = 30  # 30ms frames
frame_size = int(sample_rate * frame_duration_ms / 1000)  # Number of samples per frame
frames_per_buffer = frame_size  # Match the frame size
batch_duration = 1.0  # Batch duration in seconds for rough draft
overlap_duration = 0.5  # Overlap duration in seconds
long_batch_duration = 5.0  # Longer batch duration for refined transcription
long_batch_overlap = 2.0  # Overlap duration for longer batches
batch_size = int(sample_rate * batch_duration)
overlap_size = int(sample_rate * overlap_duration)
long_batch_size = int(sample_rate * long_batch_duration)
long_overlap_size = int(sample_rate * long_batch_overlap)
max_silence_duration = 1.0  # Maximum duration of silence before stopping transcription
max_silence_frames = int(max_silence_duration * 1000 / frame_duration_ms)
pre_speech_padding_ms = 1000  # Include 1000 ms to capture more of the initial speech
post_speech_padding_ms = 500  # Include 500 ms after silence detection to capture trailing words
pre_speech_frames = int(pre_speech_padding_ms / frame_duration_ms)
post_speech_frames = int(post_speech_padding_ms / frame_duration_ms)
energy_threshold = 20  # Adjust this value based on your environment

# Initialize VAD with a slightly less aggressive level
vad = webrtcvad.Vad(1)

# Queues and Buffers
audio_queue = queue.Queue()
transcription_queue = queue.Queue()
corrected_transcription_queue = queue.Queue()
stop_event = threading.Event()

# Ring buffer for pre-speech padding
pre_speech_buffer = collections.deque(maxlen=pre_speech_frames)

# Create a tkinter window
root = tk.Tk()
root.title("Real-Time Transcription")
root.geometry("600x400")

# Create a Text widget to display transcription
transcription_text = tk.Text(root, wrap=tk.WORD, height=20, width=70)
transcription_text.pack(padx=10, pady=10)
transcription_text.config(state=tk.DISABLED)

# Configure tags for different text colours
transcription_text.tag_configure("rough", foreground="blue")
transcription_text.tag_configure("final", foreground="black")

def update_transcription(text, timestamp):
    """Update the transcription text in the GUI with the rough draft style."""
    transcription_text.config(state=tk.NORMAL)
    transcription_text.insert(tk.END, f"{timestamp} {text}\n", ("rough",))
    transcription_text.config(state=tk.DISABLED)
    transcription_text.see(tk.END)

def append_corrected_transcription(corrected_text, timestamp):
    """Replace the transcription text with a corrected version."""
    transcription_text.config(state=tk.NORMAL)
    transcription_text.delete("1.0", tk.END)  # Clear existing rough text
    transcription_text.insert(tk.END, f"{timestamp} {corrected_text}\n", ("final",))
    transcription_text.config(state=tk.DISABLED)
    transcription_text.see(tk.END)

def audio_callback(indata, frames, time_info, status):
    """Callback function to receive audio data."""
    if status:
        logging.warning(f"Audio callback status: {status}")

    audio_data = indata.copy().flatten()  # Ensure it's 1D
    audio_queue.put(audio_data)

def audio_processing_loop():
    """Process audio frames, perform VAD, and manage batching."""
    speech_frames = []
    silence_counter = 0
    speech_counter = 0
    recording = False
    batch_buffer = np.zeros(0, dtype=np.int16)
    long_batch_buffer = np.zeros(0, dtype=np.int16)

    required_speech_frames = 5  # Number of consecutive speech frames to start recording
    required_silence_frames = max_silence_frames  # Number of consecutive silence frames to stop recording

    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_bytes = audio_data.tobytes()
        is_speech_vad = vad.is_speech(frame_bytes, sample_rate)

        # Compute the RMS energy of the frame
        mean_square = np.mean(audio_data ** 2)
        rms_energy = np.sqrt(mean_square) if mean_square > 0 else 0
        is_speech_energy = rms_energy > energy_threshold

        is_speech = is_speech_vad and is_speech_energy

        # Always add the current frame to the pre-speech buffer
        pre_speech_buffer.append(audio_data)

        if is_speech:
            silence_counter = 0
            speech_counter += 1

            if not recording and speech_counter >= required_speech_frames:
                recording = True
                speech_frames = list(pre_speech_buffer)  # Include pre-speech buffer
                batch_buffer = np.zeros(0, dtype=np.int16)
                logging.info("Speech detected. Starting transcription.")

            if recording:
                speech_frames.append(audio_data)
                batch_buffer = np.concatenate((batch_buffer, audio_data))
                long_batch_buffer = np.concatenate((long_batch_buffer, audio_data))

                if len(batch_buffer) > batch_size + overlap_size:
                    batch_buffer = batch_buffer[-(batch_size + overlap_size):]

                if len(batch_buffer) >= batch_size:
                    current_batch = batch_buffer.copy()
                    threading.Thread(target=transcribe_batch, args=(current_batch, "turbo"), daemon=True).start()
                    batch_buffer = batch_buffer[-overlap_size:]

                if len(long_batch_buffer) > long_batch_size + long_overlap_size:
                    long_batch_buffer = long_batch_buffer[-(long_batch_size + long_overlap_size):]

                if len(long_batch_buffer) >= long_batch_size:
                    current_long_batch = long_batch_buffer.copy()
                    threading.Thread(target=transcribe_batch, args=(current_long_batch, "large"), daemon=True).start()
                    long_batch_buffer = long_batch_buffer[-long_overlap_size:]
                    logging.info("Sent long batch for refined transcription.")

        else:
            speech_counter = 0
            if recording:
                silence_counter += 1
                speech_frames.append(audio_data)

                # Continue recording during post-speech padding
                if silence_counter <= post_speech_frames:
                    continue

                if silence_counter > required_silence_frames + post_speech_frames:
                    recording = False
                    silence_counter = 0
                    speech_counter = 0
                    logging.info("Silence detected. Stopping transcription.")

                    # Submit the remaining audio for a final refined transcription
                    if len(long_batch_buffer) > 0:
                        current_long_batch = np.concatenate(speech_frames).copy()
                        threading.Thread(target=transcribe_batch, args=(current_long_batch, "large"), daemon=True).start()

                    batch_buffer = np.zeros(0, dtype=np.int16)
                    long_batch_buffer = np.zeros(0, dtype=np.int16)

def transcribe_batch(audio_data, model_type):
    """Transcribe the audio batch and update the UI."""
    text_segments = process_audio(audio_data, model_type)
    if text_segments:
        if model_type == "turbo":
            for segment in text_segments:
                timestamp = segment.get('start', 0)
                text = segment.get('text', '')
                formatted_timestamp = time.strftime('%M:%S', time.gmtime(timestamp))
                transcription_queue.put((text, formatted_timestamp))
        elif model_type == "large":
            refined_text = "".join([seg.get('text', '') for seg in text_segments])
            first_timestamp = time.strftime('%M:%S', time.gmtime(text_segments[0].get('start', 0)))
            threading.Thread(target=correct_transcription, args=(refined_text, first_timestamp), daemon=True).start()

def process_audio(audio_data, model_type):
    """Function to send the recorded audio to the Groq API for transcription."""
    try:
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        wav_io.seek(0)

        audio_segment = AudioSegment.from_file(wav_io, format="wav")
        mp3_io = BytesIO()
        audio_segment.export(mp3_io, format="mp3")
        mp3_io.seek(0)

        file = ("audio.mp3", mp3_io.read())

        model = "whisper-large-v3-turbo" if model_type == "turbo" else "whisper-large-v3"

        transcription = client.audio.transcriptions.create(
            file=file,
            model=model,
            response_format="json" if model_type == "turbo" else "verbose_json",
            language="en"
        )

        if hasattr(transcription, 'text'):
            return [{"start": 0, "text": transcription.text}]
        elif hasattr(transcription, 'segments'):
            return transcription.segments
        else:
            logging.error(f"Unexpected response format: {transcription}")
            return ""

    except Exception as e:
        logging.error(f"Error during transcription with {model_type}: {e}")
        return ""

def correct_transcription(refined_text, timestamp):
    """Use the LLM to correct grammar or minor transcription errors in the refined text."""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a grammar correction assistant. Your task is to only correct any grammatical errors, "
                    "punctuation mistakes, or minor transcription errors in the provided text. "
                    "Do not include any explanations or notes. Return only the corrected version of the text. Do not include any additional text. "
                    "Do not alter the meaning or tone of the original content."
                )
            },
            {
                "role": "user",
                "content": refined_text
            }
        ]
        completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.0,
            max_tokens=1024
        )
        corrected_text = completion.choices[0].message.content.strip()
        corrected_transcription_queue.put((corrected_text, timestamp))
    except Exception as e:
        logging.error(f"Error during grammar correction: {e}")

def transcription_output_loop():
    """Continuously get transcriptions and update the GUI."""
    while not stop_event.is_set():
        try:
            if not transcription_queue.empty():
                rough_text, timestamp = transcription_queue.get(timeout=0.1)
                update_transcription(rough_text.strip(), timestamp)
                logging.info(f"Rough draft added: {rough_text.strip()} at {timestamp}")

            if not corrected_transcription_queue.empty():
                corrected_text, timestamp = corrected_transcription_queue.get(timeout=0.1)
                append_corrected_transcription(corrected_text, timestamp)
                logging.info(f"Final corrected transcription: {corrected_text}")
        except queue.Empty:
            continue

def start_listening():
    """Start the audio stream and handle callbacks."""
    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16',
                        blocksize=frames_per_buffer, callback=audio_callback):
        logging.info("Listening... Press Ctrl+C to stop.")

        audio_thread = threading.Thread(target=audio_processing_loop, daemon=True)
        transcription_thread = threading.Thread(target=transcription_output_loop, daemon=True)
        audio_thread.start()
        transcription_thread.start()

        try:
            root.mainloop()  # Start the tkinter event loop
        except KeyboardInterrupt:
            logging.info("Stopping...")
            stop_event.set()
            audio_thread.join()
            transcription_thread.join()

def main():
    start_listening()

if __name__ == '__main__':
    main()
