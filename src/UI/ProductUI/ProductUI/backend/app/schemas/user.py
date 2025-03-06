from pydantic import BaseModel
from typing import List, Optional

class UserBase(BaseModel):
    username: str
    email: str
    password: str
    favorite_teams: List[int] = [] # list of team ids
    class Config:
        orm_mode = True
