from pydantic import BaseModel
from typing import List, Optional

class UserBase(BaseModel):
    username: str
    email: str
    password: str
    favorite_teams: List[int] = [] # list of team ids
    class Config:
        orm_mode = True

class TeamInput (BaseModel):
    team_name1: str
    team_name2: str

    class Config:
        orm_mode = True