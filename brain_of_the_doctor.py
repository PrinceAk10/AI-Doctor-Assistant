# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

# Step 1: Setup GROQ API key
import os
import base64
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Error: Missing GROQ_API_KEY. Set it in the environment variables.")

# Step 2: Convert image to base64 format
def encode_image(image_path):   
    """
    Reads an image file and encodes it into a base64 string.

    Args:
    image_path (str): Path to the image file.

    Returns:
    str: Base64 encoded string of the image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    return encoded_image  # Return the correctly encoded image


# Step 3: Setup Multimodal LLM
def analyze_image_with_query(query, encoded_image, model="llama-3.2-90b-vision-preview"):
    """
    Sends an image and text query to Groq's multimodal LLM.

    Args:
    query (str): The query text.
    encoded_image (str): Base64 encoded image data.
    model (str): Model name for analysis.

    Returns:
    str: AI-generated response.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)  # Explicitly pass API key
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                ],
            }
        ]
        chat_completion = client.chat.completions.create(messages=messages, model=model)
        
        # Ensure a valid response exists
        if chat_completion and chat_completion.choices:
            return chat_completion.choices[0].message.content.strip()
        else:
            return "Error: No valid response from the model."

    except Exception as e:
        return f"Error during image analysis: {e}"


# Step 4: Symptom Checker
def symptom_checker(user_input):
    """
    Analyzes the user's input for symptoms and provides follow-up questions or a preliminary diagnosis.

    Args:
    user_input (str): The user's description of symptoms.

    Returns:
    str: Follow-up questions or preliminary diagnosis.
    """
    # Example symptom mapping (can be expanded with a more comprehensive dataset)
    symptom_mapping = {
        "headache": "Do you have any other symptoms like fever or nausea?",
        "fever": "How long have you had the fever? Is it accompanied by chills or sweating?",
        "cough": "Is your cough dry or productive? Do you have shortness of breath?",
        "chest pain": "Is the pain sharp or dull? Does it radiate to your arm or jaw?",
        "fatigue": "Have you been experiencing fatigue for a long time? Do you have trouble sleeping?",
        "abdominal pain": "Where exactly is the pain located? Is it sharp or cramping?",
    }

    # Check if any symptom keywords are present in the user's input
    for symptom, question in symptom_mapping.items():
        if symptom in user_input.lower():
            return f"I see you mentioned {symptom}. {question}"

    # If no specific symptom is found, ask a general follow-up question
    return "Can you describe your symptoms in more detail?"


# Step 5: Fetch Medical Knowledge
def fetch_medical_knowledge(query):
    """
    Fetches relevant medical knowledge from a knowledge base or API.

    Args:
    query (str): The user's query or symptom description.

    Returns:
    str: Relevant medical information.
    """
    # Example: Use a simple placeholder for now (can be replaced with an API call to PubMed or similar)
    medical_knowledge = {
        "headache": "Headaches can be caused by stress, dehydration, or migraines. Drink water and rest.",
        "fever": "Fever is often a sign of infection. Monitor your temperature and stay hydrated.",
        "cough": "A cough can be due to a cold, flu, or allergies. Rest and drink warm fluids.",
        "chest pain": "Chest pain can indicate heart issues. Seek medical attention immediately.",
        "fatigue": "Fatigue can result from lack of sleep, stress, or underlying health conditions.",
        "abdominal pain": "Abdominal pain can be caused by indigestion, gas, or more serious conditions.",
    }

    # Check if the query matches any medical knowledge
    for keyword, info in medical_knowledge.items():
        if keyword in query.lower():
            return f"Medical Information: {info}"

    # If no match is found, return a generic response
    return "I recommend consulting a healthcare professional for more detailed information."


# Example Usage
if __name__ == "__main__":
    image_path = "acne.jpg"  # Example image path
    try:
        encoded_image = encode_image(image_path)
        query = "Is there something wrong with my face?"
        response = analyze_image_with_query(query, encoded_image)
        print("AI Doctor Response:", response)

        # Test symptom checker
        user_input = "I have a headache and fever."
        symptom_response = symptom_checker(user_input)
        print("Symptom Checker Response:", symptom_response)

        # Test medical knowledge
        medical_info = fetch_medical_knowledge(user_input)
        print("Medical Knowledge:", medical_info)
    except Exception as error:
        print(error)