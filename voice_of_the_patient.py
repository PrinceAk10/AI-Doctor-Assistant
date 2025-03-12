# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

# Step 1: Setup Audio Recorder (Requires ffmpeg & portaudio)
import logging
import os
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from groq import Groq
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Records audio from the microphone and saves it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Max wait time for speech to start.
    phrase_time_limit (int): Max duration of a single phrase.
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")

            # Record audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")

            # Convert recorded audio to MP3
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")

            logging.info(f"Audio saved to {file_path}")

    except sr.WaitTimeoutError:
        logging.error("No speech detected, recording timed out.")
    except Exception as e:
        logging.error(f"An error occurred while recording: {e}")

# Step 2: Setup Speech-to-Text (STT) Model
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY is missing! Set it in the environment variables.")

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY, language="auto"):
    """
    Transcribes audio using Groq API.

    Args:
    stt_model (str): Speech-to-text model name.
    audio_filepath (str): Path to the recorded audio file.
    GROQ_API_KEY (str): API key for authentication.
    language (str): Language of the audio (default is "auto" for automatic detection).

    Returns:
    str: Transcribed text or error message.
    """
    if not os.path.exists(audio_filepath):
        logging.error("Audio file not found!")
        return "Error: Audio file not found."

    try:
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language=language
            )

        return transcription.text if transcription else "Error: No transcription result."
    
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return "Error: Failed to transcribe audio."

# Step 3: Emotion Detection
def detect_emotion(audio_filepath):
    """
    Detects the emotion from the user's voice using an emotion detection API.

    Args:
    audio_filepath (str): Path to the audio file.

    Returns:
    str: Detected emotion (e.g., "happy", "sad", "neutral").
    """
    # Example: Use a placeholder emotion detection API (replace with a real API)
    API_KEY = os.environ.get("EMOTION_API_KEY")  # Set your API key in .env
    API_URL = "https://api.example.com/emotion-detection"  # Replace with actual API URL

    if not API_KEY:
        return "neutral"  # Fallback if no API key is provided

    try:
        # Send the audio file to the emotion detection API
        with open(audio_filepath, "rb") as audio_file:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                files={"file": audio_file},
            )
        
        # Parse the API response
        if response.status_code == 200:
            emotion = response.json().get("emotion", "neutral")
            return emotion
        else:
            logging.error(f"Error: Emotion detection API returned status code {response.status_code}")
            return "neutral"
    except Exception as e:
        logging.error(f"Error during emotion detection: {e}")
        return "neutral"

# Example Usage
if __name__ == "__main__":
    # Test audio recording
    audio_filepath = "patient_voice_test.mp3"
    record_audio(file_path=audio_filepath)

    # Test transcription
    if GROQ_API_KEY:
        transcript = transcribe_with_groq(stt_model="whisper-large-v3", 
                                          audio_filepath=audio_filepath, 
                                          GROQ_API_KEY=GROQ_API_KEY,
                                          language="auto")
        logging.info(f"Transcription: {transcript}")

    # Test emotion detection
    emotion = detect_emotion(audio_filepath)
    logging.info(f"Detected Emotion: {emotion}")