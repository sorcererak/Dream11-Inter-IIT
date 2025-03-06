import requests
import json
import base64

url = "https://api.sarvam.ai/text-to-speech"
payload = {
    "inputs": ["dont know the name of the cricketer who scored 100 runs, i can only say that he is a great batsman"],
    "target_language_code": "hi-IN",
    "speaker": "amartya",
    "pitch": 0,
    "pace": 1.1,
    "loudness": 1.5,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v1"
}
headers = {
    "api-subscription-key": "",  # Your API subscription key
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    # The response is expected to be in JSON format containing base64 encoded audio
    data = response.json()  # Parse JSON response
    
    # Extract base64 audio string
    audio_data_base64 = data['audios'][0]  # Assuming the audio is the first item in the 'audios' list
    
    # Decode the base64 string to binary data
    audio_data = base64.b64decode(audio_data_base64)
    
    # Save the decoded audio data to a .wav file
    with open("output_audio.wav", "wb") as audio_file:
        audio_file.write(audio_data)
    print("Audio file has been saved as 'output_audio.wav'.")
    
else:
    print(f"Failed to generate audio. Status code: {response.status_code}")
    print("Response:", response.text)