# Real-Time Speech Transcription with Grammar Correction

## Overview
This project is a real-time speech transcription system that utilizes the Groq API to convert spoken audio into text. It uses a Voice Activity Detection (VAD) system to determine when speech starts and stops, ensuring accurate transcription of spoken words. After generating a rough transcription, the project uses a language model to correct grammatical errors and provide a more polished transcription.

The application is built using Python, leveraging the following key technologies:
- **Sounddevice**: For capturing live audio from a microphone.
- **WebRTC VAD**: To determine when speech starts and stops, allowing for more accurate transcription.
- **Groq API**: To perform the transcription using Groq's high-performance Whisper models and to refine transcriptions using an LLM (Large Language Model).
- **Pydub**: For audio conversion and handling.
- **Tkinter**: To provide a simple GUI for displaying real-time transcriptions.

## Features
- Real-time transcription of audio with timestamps.
- Grammar correction using a language model for improved text output.
- GUI to display rough and corrected transcriptions.
- Configurable parameters for speech detection sensitivity.

## Requirements
Before running this project, ensure that you have the following installed:
- Python 3.7+
- A Groq API key for accessing transcription and LLM services.

## Installation
1. Clone the repository to your local machine:
   
   git clone https://github.com/yourusername/real-time-speech-transcription.git
   cd real-time-speech-transcription

2. Install the required Python libraries:
   
   pip install -r requirements.txt

3. Set up the Groq API key:
   - Obtain your API key from the Groq platform.
   - Set the `GROQ_API_KEY` environment variable:

   For Windows:
   
   set GROQ_API_KEY=your_api_key_here

   For MacOS/Linux:
   
   export GROQ_API_KEY=your_api_key_here

## Usage
To start the real-time transcription system, simply run the following command:

   python main.py

This will open a simple GUI window that displays real-time transcriptions as you speak.

### How It Works
1. **Audio Capture**: Uses `sounddevice` to capture audio from the user's microphone in real-time.
2. **Voice Activity Detection (VAD)**: `webrtcvad` is used to detect when the user starts and stops speaking, enabling accurate detection of speech segments.
3. **Transcription**:
   - Uses the Groq API to transcribe audio segments using the `whisper-large-v3-turbo` model for quick, rough drafts.
   - Longer audio segments are transcribed with `whisper-large-v3` for more accurate, refined transcriptions.
4. **Grammar Correction**: Once the refined transcription is complete, the project uses a Groq-compatible LLM (e.g., `llama3-8b-8192`) to correct any grammatical errors in the transcription.
5. **GUI Display**: The transcriptions are displayed in a Tkinter GUI window:
   - Rough drafts are shown in blue.
   - Once a corrected transcription is ready, it replaces the rough draft in black text.

### Configuration
You can adjust the following parameters directly in the `main.py` file to better suit your environment:
- `sample_rate`: Adjust the audio sample rate (default is 16 kHz).
- `energy_threshold`: Fine-tune this to detect quieter or louder voices.
- `pre_speech_padding_ms` and `post_speech_padding_ms`: To adjust how much audio is captured before and after speech is detected.

## Known Issues
- **Latency**: Depending on the speed of your internet connection and the Groq API response times, there may be a slight delay between speaking and seeing the transcription.
- **VAD Sensitivity**: Adjust the `vad.set_mode` level and `energy_threshold` based on your environment to minimize false positives or missed speech.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

### Contact
For any questions or support, feel free to open an issue on the GitHub repository or reach out to me at varadganjoo@gmail.com.
