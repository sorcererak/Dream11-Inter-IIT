from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.schemas.pydantic_schema import PlayerStatsInput,PlayersInput, PlayerInput
from app.services.player import get_player_stats_by_name_from_db2,get_player_batting_stats_from_db, get_player_bowling_stats_from_db,get_all_player_info_for_player_ids_from_db,get_all_player_ids_played_for_team_from_db,get_player_stats_by_name_from_db,get_player_lifetime_stats_from_db,get_all_players_stats_from_db,get_player_stats_from_db,get_teams_player_stats_from_db,get_match_player_stats_from_db

router = APIRouter()

@router.get("/")
def main_function():
    return "Player Route is running......🥳!!"


@router.get("/match/{match_id}")
async def get_all_players(match_id : str , db: Session = Depends(get_db)):
    try:
        players = get_all_players_stats_from_db(db,match_id)
        return {"status": "ok", "message": "Players retrieved successfully", "data": players}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/player_stats")
async def get_player_stats(playerInput: PlayerStatsInput, db: Session = Depends(get_db)):
    try:
        player = get_player_stats_from_db(db,playerInput.match_id, playerInput.player_id)
        return {"status": "ok", "message": "Player retrieved successfully", "data": player}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cricketers_lifetime_stats/{player_id}")  
async def get_all_players(player_id: str, db: Session = Depends(get_db)):
    try:
        player = get_player_lifetime_stats_from_db(db,player_id)
        return {"status": "ok", "message": "Player retrieved successfully", "data": player}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/player_stats/all")
async def get_player_stats(playerInput: PlayersInput, db: Session = Depends(get_db)):
    try:
        player = get_teams_player_stats_from_db(db,playerInput.match_id, playerInput.player_ids)
        return {"status": "ok", "message": "Player retrieved successfully", "data": player}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/player_stats/{match_id}")
async def get_all_players(match_id : str , db: Session = Depends(get_db)):
    try:
        players = get_match_player_stats_from_db(db,match_id)
        return {"status": "ok", "message": "Players retrieved successfully", "data": players}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


   
@router.post("/player_all_stats/")
async def get_all_players(playerInput: PlayerInput, db: Session = Depends(get_db)):
    try:
        players = get_player_stats_by_name_from_db(db,playerInput.player_id,playerInput.match_id)
        return {"status": "ok", "message": "Players retrieved successfully", "data": players}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/player_all_stats2/")
async def get_all_players(playerInput: PlayerInput, db: Session = Depends(get_db)):
    try:
        players = get_player_stats_by_name_from_db2(db,playerInput.player_id,playerInput.match_id)
        return {"status": "ok", "message": "Players retrieved successfully", "data": players}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/search_players/{team_name}")
async def get_all_players(team_name : str , db: Session = Depends(get_db)):
    try:
        players_ids = get_all_player_ids_played_for_team_from_db(db,team_name)
        players = get_all_player_info_for_player_ids_from_db(db,players_ids)
        player_dict = {}
        for player in players:
            player_dict[player.player_id] = player
        return {"status": "ok", "message": "Players retrieved successfully", "players": player_dict , "player_ids": players_ids, "count": len(players)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    