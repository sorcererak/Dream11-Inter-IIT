from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.team import TeamInput
from app.schemas.pydantic_schema import ModelInput
from app.services.match import get_team_match_wins_from_db,get_pitch_type_from_db,get_all_matches_for_date_from_db,get_match_weather_from_db,get_data_from_csv,get_match_details_from_db,get_all_featured_matches_for_date_from_db,get_all_matches_from_db,get_all_team_matches_from_db,get_all_teams_matches_from_db,match_to_dict
from app.services.team import get_teams_by_name_from_db,get_team_info_by_name_from_db
from app.services.player import get_all_match_players_profile_from_db,get_player_ids_for_match, get_player_profile_for_ids
from fastapi import File, UploadFile
from ...product_ui_model.Product_UI_runner import main as get_player_scores
from fastapi.responses import JSONResponse
from app.utils.players_map import runner_main
import random
import logging
import math
from app.product_ui_model.Product_UI_runner import main as score_predictor

router = APIRouter()

@router.get("/")
def main_function():
    return "Match Route is running......🥳!!"


@router.get("/all")
async def get_all_matches(db: Session = Depends(get_db)):
    try:
        matches = get_all_matches_from_db(db)
        return {"status": "ok", "message": "Teams retrieved successfully", "data": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/date/all")
async def get_matches_by_date(date: str, db: Session = Depends(get_db)):
    try:
        matches = get_all_matches_for_date_from_db(db,date)
        for match in matches:
            teamA=match.teams[0]
            teamB=match.teams[1]
            teamA_info = get_team_info_by_name_from_db(db, teamA)
            teamB_info = get_team_info_by_name_from_db(db, teamB)
            match.team_info = {
                "teamA": teamA,
                "teamAinfo": teamA_info,
                "teamB": teamB,
                "teamBinfo": teamB_info,
            }


        return {"status": "ok", "message": "Teams retrieved successfully", "data": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/date/featured")
async def get_matches_by_date(date: str, db: Session = Depends(get_db)):
    try:
        matches = get_all_featured_matches_for_date_from_db(db,date)
        for match in matches:
            teamA=match.teams[0]
            teamB=match.teams[1]
            teamA_info = get_team_info_by_name_from_db(db, teamA)
            teamB_info = get_team_info_by_name_from_db(db, teamB)
            match.team_info = {
                "teamA": teamA,
                "teamAinfo": teamA_info,
                "teamB": teamB,
                "teamBinfo": teamB_info,
            }
        return {"status": "ok", "message": "Teams retrieved successfully", "data": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/team/{team_name}")
async def get_matches_by_team_id(team_name: str, db: Session = Depends(get_db)):
    try:
        matches = get_all_team_matches_from_db(db,team_name)
        return {"status": "ok", "message": "Teams retrieved successfully", "data": matches}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/team")
async def get_matches_by_team_id(teams: TeamInput, db: Session = Depends(get_db)):
    try:
        db.expunge_all()
        if teams.team_name1 == teams.team_name2:
            raise HTTPException(status_code=400, detail="Both teams cannot be same")
        matches = get_all_teams_matches_from_db(db,teams.team_name1, teams.team_name2)
        team_info = get_teams_by_name_from_db(db,teams.team_name1, teams.team_name2)
        match_dicts = [match_to_dict(match) for match in matches]
        return {"status": "ok", "message": "Teams retrieved successfully", "data": match_dicts, "team_info": team_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/matchdetails/{match_id}")
async def get_match_details(match_id: str, db: Session = Depends(get_db)):
    try:
        rows = get_player_ids_for_match(db, match_id)
        player_ids = [row[0] for row in rows]
        players = get_player_profile_for_ids(db, player_ids)
        matchdetails = get_match_details_from_db(db, match_id)
        
        pitch_type = get_pitch_type_from_db(db, matchdetails.venue)
        wins = get_team_match_wins_from_db(db, matchdetails.teams[0], matchdetails.teams[1], matchdetails.dates[0])
        # Fetch team info for valid teams
        team_info = {}
        team_info["teamA"] = get_team_info_by_name_from_db(db, matchdetails.teams[0])
        team_info["teamB"] = get_team_info_by_name_from_db(db, matchdetails.teams[1])
        teamA = []
        teamB = []
        for player in players:
            if player.unique_name in matchdetails.players:
                teamA.append(player)
            else:
                teamB.append(player)

        return {
            "status": "ok",
            "message": "Data retrieved successfully",
            "matchdetails": matchdetails,
            "player_count": len(teamA) + len(teamB),
            "player_ids": player_ids,
            "teamA": teamA,
            "teamB": teamB,
            "pitch": pitch_type,
            "wins"  : wins,
            "team_info": team_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dreamTeam/{match_id}")
async def dreamScores(match_id: str, db: Session = Depends(get_db)):
    try:
        # Fetch player IDs for the given match
        rows = get_player_ids_for_match(db, match_id)
        player_ids = [row[0] for row in rows]
        match_details = get_match_details_from_db(db, match_id)
        players = get_player_profile_for_ids(db, player_ids)

        match_date = match_details.dates[0]
        players_with_points = assign_dream_points(players, match_date, match_details.match_type)

        return {
            "status": "ok",
            "message": "Teams retrieved successfully",
            "match_details": match_details,
            "count": len(players_with_points),
            "players": players_with_points,
            "pitch": "Grass"
        }

    except Exception as e:
        logging.error(f"Error fetching dream scores: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching dream scores.")
# @router.post("/dreamTeam")
# async def dreamScores(match_id: int, db: Session = Depends(get_db)):
#     try:
#         return {"status": "ok", "message": "Teams retrieved successfully", "data": [DreamTeam,DreamTeam,DreamTeam]}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/dreamTeam")
async def dreamScores(modelInput: ModelInput, db: Session = Depends(get_db)):
    try:
        players = get_player_profile_for_ids(db, modelInput.player_ids)
        print("Players Fetched", players)
    except Exception as e:
        logging.error(f"Error fetching players api: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching players.")
    try:
        players_with_points = assign_dream_points(players, modelInput.match_date, modelInput.match_type)
        print("Players Assigned Points", players_with_points)
        return {
                "status": "ok",
                "message": "Teams retrieved successfully",
                "count": len(players_with_points),
                "players": players_with_points
            }
    except Exception as e:
        logging.error(f"Error fetching dream scores api: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching dream scores.")
    

@router.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Step 1: Read and process the uploaded CSV
        data = await get_data_from_csv(file)
        
        # Step 2: Run the runner_main function to process the data and extract player info
        try:
            result_df = runner_main(data)
            print(result_df)  # Assuming this returns a DataFrame with player_id and player_team
        except Exception as e:
            logging.error(f"Error running runner_main: {e}")
            raise HTTPException(status_code=500, detail="Error running runner_main")
        
        # Step 3: Get the list of player_ids from result_df
        try:
            player_ids = result_df['player_id'].tolist()
            # Remove NaN values from player_ids
            player_ids = [player_id for player_id in player_ids if not (isinstance(player_id, float) and math.isnan(player_id))]
            print(player_ids)
        except Exception as e:
            logging.error(f"Error getting player_ids: {e}")
            raise HTTPException(status_code=500, detail="Error getting player_ids")
        
        # Step 4: Get player profiles from the database
        try:
            players = get_player_profile_for_ids(db, player_ids)
        except Exception as e:
            logging.error(f"Error fetching player profiles: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
        # Step 5: Create a mapping of player_id to player_team
        try:
            player_team_map = dict(zip(result_df['player_id'], result_df['player_team']))
        except Exception as e:
            logging.error(f"Error creating player_team map: {e}")
            raise HTTPException(status_code=500, detail="Error creating player_team map")

        # Step 6: Add player_team to each player object and convert them to dictionaries
        players_with_team = []
        for player in players:
            player_dict = player.__dict__  # Convert the object to a dictionary
            player_dict["player_team"] = player_team_map.get(player.player_id, None)
            players_with_team.append(player_dict)

        # Step 7: Get unique teams, filtering out 'NaN' values
        try:
            unique_teams = result_df['player_team'].dropna().unique()  # Drop NaN values
            print("Unique Teams:", unique_teams)

            if len(unique_teams) != 2:
                raise HTTPException(status_code=400, detail="Only Provide two teams in the data")

            match_date = data['Match Date'].iloc[0]
            match_type = data['Format'].iloc[0]
            teamA_players = [player for player in players_with_team if player["player_team"] == unique_teams[0]]
            teamB_players = [player for player in players_with_team if player["player_team"] == unique_teams[1]]
            
            # Fetch team info for valid teams
            team_info = {}
            team_info["teamA"] = get_team_info_by_name_from_db(db, unique_teams[0])
            team_info["teamB"] = get_team_info_by_name_from_db(db, unique_teams[1])
        except Exception as e:
            logging.error(f"Error fetching team information: {e}")
            raise HTTPException(status_code=500, detail="Error fetching team information")
        
        # Step 8: Return the response with players divided into teamA and teamB arrays, and team info
        return {
            "status": "ok",
            "message": "Data retrieved successfully",
            "match_date": match_date,
            "match_type": match_type,
            "teamA": teamA_players,  # Players for Team A
            "teamB": teamB_players,  # Players for Team B
            "player_count": len(players_with_team),
            "team_info": team_info
        }

    except Exception as e:
        logging.error(f"Error in upload_csv: {e}")
        raise HTTPException(status_code=400, detail="Error in upload_csv")

@router.get("/weather/{match_id}")
async def get_weather(match_id: str, db: Session = Depends(get_db)):
    try:
        match = get_match_weather_from_db(db, match_id)
        return {"status": "ok", "message": "Weather retrieved successfully", "data": match}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def assign_dream_points(players, match_date, match_type):
    # Simulate retrieving dream points
    try:
        print("Calling get_player_scores")
        for player in players:
            print(player.player_id)
        print("Match Date", match_date)
        print("Match Type", match_type)
        dream_points = get_player_scores(match_type, [player.player_id for player in players], match_date)
    except Exception as e:
        logging.error(f"Error fetching dream points in assign function : {e}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching dream points.")
    
    # Map the dream_points by player_id
    try:
        players_map = {player['player_id']: player for player in dream_points}
        # print(players_map)

    except Exception as e:
        logging.error(f"Error mapping dream points: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while mapping dream points.")
    

    # Assign predicted_score to players based on the dream_points
    try:
        for player in players:
            if player.player_id in players_map:  
                if (players_map[player.player_id]["predicted_score"]) > 500:
                    player.predicted_score = 112
                else:
                    player.predicted_score = players_map[player.player_id]["predicted_score"]  # Use dot notation here too
    except Exception as e:
        logging.error(f"Error assigning dream points: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while assigning dream points.")

    # Return the updated players list
    return players
