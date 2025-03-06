import requests
import base64
from fastapi import HTTPException
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

API_SUBSCRIPTION_KEY = os.getenv('API_SUBSCRIPTION_KEY')
TEXT_TO_SPEECH_API_URL = os.getenv('TEXT_TO_SPEECH_API_URL')
TRANSLATE_API_URL = "https://api.sarvam.ai/translate"

def translate_text(text: str, target_language_code: str, source_language_code: str) -> str:
    """
    Translate text from the source language to the target language using the translation API.
    """
    payload = {
        "input": text,
        "target_language_code": target_language_code,
        "source_language_code": source_language_code
    }

    headers = {
        "api-subscription-key": API_SUBSCRIPTION_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(TRANSLATE_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json().get('translated_text', '')
    else:
        raise HTTPException(status_code=500, detail=f"Translation failed: {response.text}")


def convert_text_to_audio(text: str, target_language_code: str) -> bytes:
    """
    Convert text to audio in the specified language. Handles splitting large texts into smaller chunks.
    """
    try:
        # Step 1: Translate the text to the target language
        if target_language_code == "en-IN":
            # Skip translation if the target language is English (India)
            translated_text = text
        else:
            translated_text = translate_text(text, target_language_code, "en-IN")
        print(f"Translated Text: {translated_text}")

        # Step 2: Split the translated text into smaller chunks
        text_chunks = split_text(translated_text)
        print(f"Text split into {len(text_chunks)} chunks")

        # Step 3: Initialize an empty list to hold audio data for each chunk
        audio_chunks = []

        # Step 4: Generate audio for each chunk
        for idx, chunk in enumerate(text_chunks):
            print(f"Generating audio for chunk {idx+1}/{len(text_chunks)}")
            try:
                print(f"Chunk {idx+1}: {chunk}")
                audio_data = generate_audio_for_chunk(chunk, target_language_code)
                print(f"Audio generated for chunk {idx+1}, size: {len(audio_data)} bytes")
                audio_chunks.append(audio_data)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to generate audio for chunk {idx+1}: {str(e)}")
        
        # Step 5: Combine all audio chunks into one final audio file
        final_audio_data = b''.join(audio_chunks)
        print(f"Final combined audio size: {len(final_audio_data)} bytes")
        return final_audio_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def generate_audio_for_chunk(chunk: str, target_language_code: str) -> bytes:
    """
    Generate audio for a single chunk of text using the Sarvam AI Text-to-Speech API.
    """
    try:
        url = TEXT_TO_SPEECH_API_URL
        payload = {
            "inputs": [chunk],
            "target_language_code": target_language_code,
            "speaker": "amartya",  # Speaker name can be dynamic if needed
            "pitch": 0,
            "pace": 1.1,
            "loudness": 1.5,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": "bulbul:v1"
        }

        headers = {
            "api-subscription-key": API_SUBSCRIPTION_KEY,  # API subscription key
            "Content-Type": "application/json"
        }

        # Send the chunk to the TTS API for audio generation
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            # Extract the base64 encoded audio and decode it to bytes
            data = response.json()
            audio_data_base64 = data['audios'][0]
            audio_data = base64.b64decode(audio_data_base64)
            return audio_data
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio for chunk")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while generating audio for chunk: {str(e)}")


def split_text(text: str, max_length: int = 500) -> list:
    """
    Split the text into smaller chunks to avoid exceeding API limits (max_length).
    """
    # Split text by spaces to avoid cutting words in the middle
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(' '.join(chunk)) > max_length:
            chunks.append(' '.join(chunk[:-1]))  # Add the previous chunk
            chunk = [word]  # Start a new chunk with the current word

    # Add the last chunk if there are any remaining words
    if chunk:
        chunks.append(' '.join(chunk))
    
    return chunks

