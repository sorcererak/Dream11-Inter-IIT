import requests
import base64
from dotenv import load_dotenv
import os
from fastapi import HTTPException

# Load environment variables from the .env file
load_dotenv()

# Retrieve values from environment variables
API_SUBSCRIPTION_KEY = os.getenv('API_SUBSCRIPTION_KEY')
TEXT_TO_SPEECH_API_URL = os.getenv('TEXT_TO_SPEECH_API_URL')
TRANSLATE_API_URL = "https://api.sarvam.ai/translate"  # Sarvam AI translation endpoint
def translate_text(text: str, target_language_code: str, source_language_code: str) -> str:
    """
    Function to translate text to the desired target language using Sarvam AI API
    """
    payload = {
        "input": text,  # Use 'input' instead of 'text' as per API requirements
        "target_language_code": target_language_code,
        "source_language_code": source_language_code  # Add source language
    }

    headers = {
        "api-subscription-key": API_SUBSCRIPTION_KEY,  # Your API key from environment variable
        "Content-Type": "application/json"
    }

    # Send a request to the translation API
    response = requests.post(TRANSLATE_API_URL, json=payload, headers=headers)

    print(f"Translation API Response Status: {response.status_code}")
    print(f"Translation API Response: {response.text}")  # Print the response content for debugging

    if response.status_code == 200:
        data = response.json()
        return data['translated_text']
    else:
        raise HTTPException(status_code=500, detail=f"Translation failed: {response.text}")

def convert_text_to_audio(text: str, target_language_code: str) -> bytes:
    """
    Function to convert text to audio in the specified language
    """
    try:
        # Translate text to the target language using Sarvam AI
        try:
            translated_text = translate_text(text, target_language_code, "en-IN")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
        
        # Send the translated text to the Sarvam AI TTS API
        url = TEXT_TO_SPEECH_API_URL 
        
        payload = {
            "inputs": [translated_text],
            "target_language_code": target_language_code,
            "speaker": "amartya",  # You can make this dynamic based on user preference
            "pitch": 0,
            "pace": 1.1,
            "loudness": 1.5,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": "bulbul:v1"
        }

        headers = {
            "api-subscription-key": API_SUBSCRIPTION_KEY,  # Your API key
            "Content-Type": "application/json"
        }

        # Send the translated text to the TTS API
        response = requests.post(url, json=payload, headers=headers)

        # print(f"TTS API Response Status: {response.status_code}")
        # print(f"TTS API Response: {response.text}")  # Print the response content for debugging

        if response.status_code == 200:
            # Extract base64 audio and decode to bytes
            data = response.json()
            audio_data_base64 = data['audios'][0]
            audio_data = base64.b64decode(audio_data_base64)
            return audio_data
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
    except Exception as e:
        # Return the error message properly using 'detail'
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
