from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, validator
from io import BytesIO
from fastapi.responses import StreamingResponse
# from ...product_ui_model.Product_UI_runner import main as get_player_scores
from ...chatbot.chat_bot import tool_selector
from ...utils.translate_speech_to_text import convert_text_to_audio 
from typing import List, Union, Literal
import datetime
import re
import pandas as pd
from ...product_ui_model.description.LLM_team_explainer import add_performance_parameters
# Initialize FastAPI and Router
router = APIRouter()

# Predefined match files (you can extend this as needed)

# Pydantic model for the chat message
class ChatMessage(BaseModel):
    message: str
class ModelInput(BaseModel):
    match_type: str
    player_ids: Union[str, List[str]]
    match_date: str
    @validator('match_date')
    def validate_match_date_format(cls, v):
        # Regular expression for YYYY-MM-DD format
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(date_pattern, v):
            raise ValueError("Invalid match_date format. It must be in 'YYYY-MM-DD' format.")
        try:
            # Attempt to convert to datetime
            pd.to_datetime(v)
        except Exception as e:
            raise ValueError(f"Invalid match_date value: {v}. Unable to parse date. Error: {e}")
        return v
class AudioInput(BaseModel):
    message: str
    target_language_code: str


class DescriptionInput(BaseModel):
    match_type: str
    player_ids: Union[str, List[str]]
    
    @validator('match_type')
    def validate_match_type(cls, v):
        # Convert the match_type to lowercase
        match_files = ['odi', 't20', 'test', "odm","it20","mdm"]
        
        match_type = v.lower()
        print(f"Processed match_type: {match_type}")  # Debugging line to check the value
        # Check if the match_type is in the valid match files
        if match_type not in match_files:
            raise ValueError("Invalid match type.")
        return match_type
    
    
    @validator('player_ids', pre=True)
    def validate_player_ids(cls, v):
        if not v:
            raise ValueError("Player IDs are required") 
        if len(v) != 11:
            raise ValueError("Exactly 11 Player IDs are required")
        return v
# Chat endpoint to return response from chatbot tool_selector
@router.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
        # chat_message.message = "how to create team"
        response = tool_selector(chat_message.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Audio endpoint to convert text to audio and return the audio file
@router.post("/audio")
def text_to_speech(AudioInput: AudioInput):
    """
    Endpoint to handle text-to-speech conversion.
    """
    try:
        audio_data = convert_text_to_audio(AudioInput.message, AudioInput.target_language_code)
        
        # Create an in-memory file-like object from the audio data
        audio_file = BytesIO(audio_data)
        audio_file.seek(0)

        # Return the audio as a streaming response
        return StreamingResponse(audio_file, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=audio.wav"})

    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# @router.post("/predict")
# async def predict_scores(modelInput :ModelInput ):
# # async def predict_scores(match_type: str, player_ids: Union[str, List[str]], match_date: str):
#     # match_date="2024-12-01"
#     # player_ids = ["0085a7ce", "00823a96"]
#     # match_type = 'Test'
#     players = get_player_scores(modelInput.match_type, modelInput.player_ids, modelInput.match_date)
#     # return {"player_scores": players, "count": len(players)}
#     return players

@router.post("/description")
async def get_description(DescriptionInput :DescriptionInput):
    try:
        response = add_performance_parameters(DescriptionInput.match_type, DescriptionInput.player_ids)
        print(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))