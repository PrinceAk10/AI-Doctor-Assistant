# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# VoiceBot UI with Gradio
import os
import gradio as gr
import pygame
from deep_translator import GoogleTranslator
from gtts import gTTS
from elevenlabs.client import ElevenLabs
import time

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq

# Step 1: Set Up API Key
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY is missing. Set it as an environment variable or define it directly.")

# Initialize ElevenLabs client
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Supported languages for gTTS
GTTS_SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Urdu": "ur",
    "Gujarati": "gu",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Odia": "or",
    "Punjabi": "pa",
    "Assamese": "as",
    "Maithili": "hi",  # No direct support, using Hindi
    "Santali": "hi"  # No direct support, using Hindi
}

# System prompt for AI response
system_prompt = """You have to act as a professional doctor. 
            What's in this image? Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Do not add any numbers or special characters 
            in your response. Your response should be in one long paragraph. Always answer as if you are speaking 
            to a real person. Do not say 'In the image I see' but say 'With what I see, I think you have ...' 
            Do not respond as an AI model or in markdown, your answer should mimic that of an actual doctor. 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away."""


def text_to_speech_with_gtts(input_text, output_filepath, lang="en"):
    """
    Convert text to speech using gTTS (Google Text-to-Speech).
    """
    lang_code = GTTS_SUPPORTED_LANGUAGES.get(lang, "en")  # Default to English if not found
    temp_filepath = f"{output_filepath}_{int(time.time())}.mp3"  # Generate unique filename
    audioobj = gTTS(text=input_text, lang=lang_code, slow=False)
    audioobj.save(temp_filepath)
    play_audio(temp_filepath)
    return temp_filepath  # Return the new file path


def text_to_speech_with_elevenlabs(input_text, output_filepath):
    """
    Convert text to speech using ElevenLabs API.
    """
    try:
        input_text = input_text.encode("utf-8").decode("utf-8")  # Ensure UTF-8 encoding
        print(f"Generating audio for text: {input_text}")
        
        # Generate audio using ElevenLabs API
        audio_stream = client.generate(
            text=input_text,
            voice="Aria",  # You can change the voice as needed
            model="eleven_turbo_v2"  # Use the desired model
        )
        
        temp_filepath = f"{output_filepath}_{int(time.time())}.mp3"  # Generate unique filename
        
        # Check if audio_stream is a generator
        if hasattr(audio_stream, "__iter__"):
            print("Audio stream is a generator. Writing to file...")
            with open(temp_filepath, "wb") as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)  # Write each chunk to the file
            print(f"Audio saved to {temp_filepath}")
        else:
            print("No audio data received from ElevenLabs API.")
            return None
        
        # Play the audio
        play_audio(temp_filepath)
        return temp_filepath  # Return the new file path
    except Exception as e:
        print(f"Error using ElevenLabs API: {e}")
        return None


def play_audio(output_filepath):
    """
    Play the audio file using pygame.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(output_filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait for the audio to finish playing
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()  # Unload mixer after playback
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")


def process_inputs(audio_filepath, image_filepath, language):
    """Processes user input, transcribes audio, analyzes image, translates response, and generates speech output."""
    
    # Ensure the audio file exists
    if audio_filepath is None or not os.path.exists(audio_filepath):
        return "No audio file provided", "No doctor response", None

    # Transcribe audio
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return "Error: Missing GROQ API Key", "No doctor response", None

    speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=GROQ_API_KEY, 
                                                 audio_filepath=audio_filepath,
                                                 stt_model="whisper-large-v3")

    # Handle image analysis
    doctor_response = "No image provided for me to analyze."
    if image_filepath and os.path.exists(image_filepath):
        encoded_image = encode_image(image_filepath)
        doctor_response = analyze_image_with_query(query=system_prompt + " " + speech_to_text_output,
                                                   encoded_image=encoded_image,
                                                   model="llama-3.2-11b-vision-preview")

    # Translate and generate speech output
    if language != "English":
        doctor_response = GoogleTranslator(source='auto', target=language.lower()).translate(doctor_response)
        audio_file = text_to_speech_with_gtts(input_text=doctor_response, output_filepath="final", lang=language)
    else:
        audio_file = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath="final")

    return speech_to_text_output, doctor_response, audio_file

# Create the Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath"),
        gr.Dropdown(choices=list(GTTS_SUPPORTED_LANGUAGES.keys()), label="Language", value="English")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response")
    ],
    title="AI Doctor with Vision and Voice",
    description="This AI doctor can analyze speech and images to provide a medical assessment in multiple languages."
)

iface.launch(debug=True)
