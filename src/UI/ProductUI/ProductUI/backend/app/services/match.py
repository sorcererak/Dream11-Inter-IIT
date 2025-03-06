from sqlalchemy.orm import Session
from app import model
from sqlalchemy import func
from sqlalchemy.sql import text
from sqlalchemy import or_
from fastapi import File, UploadFile
import pandas as pd
from io import StringIO
from sqlalchemy import text
from fastapi import HTTPException


def get_all_matches_from_db(db: Session):
    return db.query(model.Match).all()

def get_all_team_matches_from_db(db: Session, team_name: str):
    return db.query(model.Match).filter(model.Match.batting_team == team_name).all()


def get_all_teams_matches_from_db(db: Session, team1_name: str, team2_name: str):
    return db.query(model.Match).filter(
        text("teams @> ARRAY[:team1_name, :team2_name]::TEXT[]")
    ).params(team1_name=team1_name, team2_name=team2_name).all()
# def get_all_teams_matches_from_db(db: Session, team1_name: str, team2_name: str):
#     return db.query(model.Match).filter(
#         model.Match.teams.op('@>')([team1_name, team2_name])  # Checks for array overlap
#     ).all()
# Assuming model.Match is your SQLAlchemy model

def get_all_matches_for_date_from_db(db: Session, date: str):
    return db.query(model.Match).filter(
        model.Match.dates.any(date)
    ).distinct(model.Match.match_id).all()


def get_all_featured_matches_for_date_from_db(db: Session, date: str):
    # Use distinct to ensure we get only one row per match_id
    return db.query(model.Match).filter(
        model.Match.dates.any(date),
        model.Match.isfeatured == 'yes'
    ).distinct(model.Match.match_id).all()

def match_to_dict(match):
    """
    Converts a SQLAlchemy Match object to a dictionary.
    """
    return {
        "match_id": match.match_id,
        "innings": match.innings,
        "batting_team": match.batting_team,
        "city": match.city,
        "dates": match.dates,
        "event_name": match.event_name,
        "match_number": match.match_number,
        "gender": match.gender,
        "match_type": match.match_type,
        "match_referees": match.match_referees,
        "tv_umpires": match.tv_umpires,
        "umpires": match.umpires,
        "team_type": match.team_type,
        "teams": match.teams,
        "venue": match.venue,
        "players": match.players,
        "season": match.season,
    }


def get_match_details_from_db(db: Session, match_id: str):
    return db.query(model.Match).filter(model.Match.match_id == match_id).first()


async def get_data_from_csv(file: UploadFile = File(...)):
    # Read the uploaded file content
    contents = await file.read()
    # Convert bytes to string (assuming the file is UTF-8 encoded)
    csv_content = contents.decode("utf-8")
    
    # Using StringIO to simulate a file object for pandas to read the CSV
    csv_file = StringIO(csv_content)
    
    # Read CSV data into a pandas DataFrame
    df_input = pd.read_csv(csv_file)
    
    return df_input

def get_match_weather_from_db(db: Session, match_id: str):
    return db.query(model.WeatherData).filter(model.WeatherData.match_id == match_id).first().weather

def get_pitch_type_from_db(db: Session, venue: str):
    """
    Fetch the pitch type for a match by performing full-text search on the stadium name.
    """
    try:
        # Perform full-text search
        pitch = db.query(model.Stadium).filter(
            func.to_tsvector('english', model.Stadium.Name).match(venue)
        ).first()
        
        if pitch:
            return pitch.Pitch_Type
        else:
            return "Grass"
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error retrieving pitch type: " + str(e))
    
    
def get_team_match_wins_from_db(db: Session, teamA: str, teamB: str, date: str):
    return db.query(model.TeamWin).filter(
        (
            (model.TeamWin.team_a == teamA) & (model.TeamWin.team_b == teamB)
        ) | (
            (model.TeamWin.team_a == teamB) & (model.TeamWin.team_b == teamA)
        ),
        model.TeamWin.match_date == date
    ).first()