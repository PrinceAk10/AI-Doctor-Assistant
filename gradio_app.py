# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# VoiceBot UI with Gradio
import os
import gradio as gr
from datetime import datetime

# Custom modules
from brain_of_the_doctor import encode_image, analyze_image_with_query, symptom_checker, fetch_medical_knowledge
from voice_of_the_patient import record_audio, transcribe_with_groq, detect_emotion
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs, translate_text

# System prompt for AI response
system_prompt = """You have to act as a professional doctor. 
            What's in this image? Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Do not add any numbers or special characters 
            in your response. Your response should be in one long paragraph. Always answer as if you are speaking 
            to a real person. Do not say 'In the image I see' but say 'With what I see, I think you have ...' 
            Do not respond as an AI model or in markdown, your answer should mimic that of an actual doctor. 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away."""

# Memory for conversation context
conversation_memory = []

def process_inputs(audio_filepath, image_filepath, text_input, language_preference):
    """Processes user input, transcribes audio, analyzes image, and generates speech output."""
    
    global conversation_memory

    # Ensure at least one input is provided
    if audio_filepath is None and image_filepath is None and text_input.strip() == "":
        return "No input provided", "No doctor response", None

    # Transcribe audio if provided
    if audio_filepath and os.path.exists(audio_filepath):
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            return "Error: Missing GROQ API Key", "No doctor response", None

        speech_to_text_output = transcribe_with_groq(GROQ_API_KEY=GROQ_API_KEY, 
                                                     audio_filepath=audio_filepath,
                                                     stt_model="whisper-large-v3",
                                                     language="auto")
        # Detect emotion from audio
        emotion = detect_emotion(audio_filepath)
    else:
        speech_to_text_output = ""
        emotion = "neutral"

    # Combine text inputs
    user_input = text_input if text_input else speech_to_text_output

    # Add user input to conversation memory
    conversation_memory.append({"role": "user", "content": user_input})

    # Handle image analysis if provided
    if image_filepath and os.path.exists(image_filepath):
        encoded_image = encode_image(image_filepath)
        image_analysis = analyze_image_with_query(query=system_prompt, encoded_image=encoded_image,
                                                  model="llama-3.2-11b-vision-preview")
    else:
        image_analysis = ""

    # Symptom checker and triage
    symptom_analysis = symptom_checker(user_input)
    if symptom_analysis:
        conversation_memory.append({"role": "system", "content": symptom_analysis})

    # Fetch medical knowledge (RAG)
    medical_info = fetch_medical_knowledge(user_input)
    if medical_info:
        conversation_memory.append({"role": "system", "content": medical_info})

    # Generate AI response with context
    ai_response = analyze_image_with_query(query=system_prompt + " " + user_input,
                                           encoded_image=encoded_image if image_filepath else None,
                                           model="llama-3.2-11b-vision-preview",
                                           memory=conversation_memory)

    # Add AI response to conversation memory
    conversation_memory.append({"role": "assistant", "content": ai_response})

    # Translate response if needed
    if language_preference != "en":
        ai_response = translate_text(ai_response, target_lang=language_preference)

    # Convert AI response to speech
    try:
        voice_of_doctor = text_to_speech_with_elevenlabs(input_text=ai_response, output_filepath="final.mp3", language=language_preference)
    except Exception as e:
        print(f"Error in TTS: {e}")
        voice_of_doctor = None  # Fallback to text if TTS fails

    return user_input, ai_response, voice_of_doctor


# Create the Gradio interface with PWA enabled
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Record Your Voice"),
        gr.Image(type="filepath", label="Upload an Image"),
        gr.Textbox(label="Type Your Symptoms or Questions"),
        gr.Dropdown(choices=["en", "es", "fr", "hi", "zh"], value="en", label="Select Response Language")
    ],
    outputs=[
        gr.Textbox(label="Your Input"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response")
    ],
    title="AI Doctor with Advanced Features",
    description="This AI doctor can analyze speech, text, and images to provide a medical assessment. It supports multiple languages and offers advanced features like symptom checking and emotional support."
)

# Launch the application with PWA enabled
iface.launch(debug=True, share=True, pwa=True)

