from pydantic import BaseModel, validator
from typing import List, Optional
from typing import List, Union
import re
import pandas as pd

class PlayerStatsInput(BaseModel):
    match_id: str 
    player_id: str

class PlayersInput(BaseModel):
    match_id: str
    player_ids: List[str]

class PlayerInput(BaseModel):
    player_id: str
    match_id: Optional[int] = None

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