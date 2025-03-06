from sqlalchemy.orm import Session
from app import model
from sqlalchemy import func
from sqlalchemy import text
from typing import Optional



def get_all_players_stats_from_db(db: Session , match_id: str):
    return db.query(model.PlayerStats).filter(model.PlayerStats.match_id == match_id).all()

def get_player_stats_from_db(db: Session, match_id: str, player_id: str):
    return db.query(model.PlayerStats).filter(model.PlayerStats.match_id == match_id, model.PlayerStats.player_id == player_id).first()

def get_teams_player_stats_from_db(db: Session, match_id: str, player_ids: list):
    return db.query(model.PlayerStats).filter(model.PlayerStats.match_id == match_id, model.PlayerStats.player_id.in_(player_ids)).all()

def get_match_player_stats_from_db(db: Session, match_id: str):
    return db.query(model.PlayerStats).filter(model.PlayerStats.match_id == match_id).all()

def get_all_players_profile_from_db(db: Session):
    return db.query(model.Player).all()

def get_player_ids_for_match(db: Session, match_id: str):
    return db.query(model.PlayerStats.player_id).filter(model.PlayerStats.match_id == match_id).distinct().all()

def get_player_profile_for_ids(db: Session, player_ids: list):
    return db.query(model.Player).filter(model.Player.player_id.in_(player_ids)).all()

def get_all_match_players_profile_from_db(db: Session, match_id: str):
    return db.query(model.Player).filter(model.Player.match_id == match_id).all()

def get_player_lifetime_stats_from_db(db: Session, player_id: str):
    return db.query(model.CricketersLifetimeStats).filter(model.CricketersLifetimeStats.identifier == player_id).first()

def get_all_player_ids_played_for_team_from_db(db: Session, team_name: str):
    """
    Fetch all player IDs where the given team name is in the teams array.

    :param db: SQLAlchemy Session object.
    :param team_name: The name of the team to filter players by.
    :return: List of player IDs.
    """
    query = text("SELECT player_id FROM team_played WHERE :team_name = ANY(teams)")
    result = db.execute(query, {"team_name": team_name}).fetchall()
    return [row[0] for row in result]  # Extract player IDs from the result

def get_all_player_info_for_player_ids_from_db(db: Session, player_ids: list):
    """
    Fetch all player information for the given player IDs.

    :param db: SQLAlchemy Session object.
    :param player_ids: List of player IDs to filter by.
    :return: List of player objects.
    """
    return db.query(model.Player).filter(model.Player.player_id.in_(player_ids)).all()

def get_player_batting_stats_from_db(db: Session, player_id: str):
    """
    Fetch batting statistics for the given player ID.

    :param db: SQLAlchemy Session object.
    :param player_id: The player ID to filter by.
    :return: Batting statistics object.
    """
    return db.query(model.PlayerBattingStats).filter(model.PlayerBattingStats.player_id == player_id).first()
def get_player_bowling_stats_from_db(db: Session, player_id: str):
    """
    Fetch bowling statistics for the given player ID.

    :param db: SQLAlchemy Session object.
    :param player_id: The player ID to filter by.
    :return: Bowling statistics object.
    """
    return db.query(model.PlayerBowlingStats).filter(model.PlayerBowlingStats.player_id == player_id).first()

def get_player_stats_by_name_from_db2(db: Session, player_id: str, match_id: Optional[int]):
    query = db.query(model.PlayerLifetimeStats)
    
    # Apply filters based on match_id
    if match_id:
        # Join with PlayerStats table if match_id is provided
        query = query.join(
            model.PlayerStats,
            model.PlayerLifetimeStats.player_id == model.PlayerStats.player_id
        ).filter(model.PlayerStats.match_id == match_id)
    
    # Join with PlayerBattingStats table
    query = query.join(
        model.PlayerBattingStats,
        model.PlayerLifetimeStats.player_id == model.PlayerBattingStats.player_id
    )
    
    # Join with PlayerBowlingStats table
    query = query.join(
        model.PlayerBowlingStats,
        model.PlayerLifetimeStats.player_id == model.PlayerBowlingStats.player_id
    )
    custom_query = db.query(model.CricketersLifetimeStats).filter(model.CricketersLifetimeStats.identifier == player_id).first()
    # Apply player_id filter
    query = query.filter(model.PlayerLifetimeStats.player_id == player_id)
    
    # Get the result
    result = query.first()
    
    if result:
        # Extract CricketersLifetimeStats data
        cricketer_data = result
        player_stats = {key: value for key, value in cricketer_data.__dict__.items() if not key.startswith('_')}
        
        # Extract PlayerStats data if match_id is provided
        if match_id:
            player_data = db.query(model.PlayerStats).filter(
                model.PlayerStats.player_id == cricketer_data.player_id,
                model.PlayerStats.match_id == match_id
            ).first()
            if player_data:
                player_dict = {key: value for key, value in player_data.__dict__.items() if not key.startswith('_')}
            else:
                player_dict = {}
        else:
            player_dict = {}
        
        # Extract PlayerBattingStats data
        batting_data = db.query(model.PlayerBattingStats).filter(
            model.PlayerBattingStats.player_id == cricketer_data.player_id
        ).first()
        if batting_data:
            batting_dict = {key: value for key, value in batting_data.__dict__.items() if not key.startswith('_')}
        else:
            batting_dict = {}
        
        # Extract PlayerBowlingStats data
        bowling_data = db.query(model.PlayerBowlingStats).filter(
            model.PlayerBowlingStats.player_id == cricketer_data.player_id
        ).first()
        if bowling_data:
            bowling_dict = {key: value for key, value in bowling_data.__dict__.items() if not key.startswith('_')}
        else:
            bowling_dict = {}
        
        # Transform batting and bowling data
        transformed_batting_data = transform_batting_data(cricketer_data, batting_dict)
        transformed_bowling_data = transform_bowling_data(cricketer_data, bowling_dict)
        
        # Combine and return the data
        return {
            "cricketer_data": {
                "player_id": cricketer_data.player_id,
                "unique_name":custom_query.unique_name,
                "full_name":custom_query.full_name,
                "player_role":custom_query.playing_role,
                "bowling_style":custom_query.bowling_style,
                "batting_style":custom_query.batting_style,
                },
            "player_data": player_dict,
            "bat": transformed_batting_data,
            "bowl": transformed_bowling_data
        }
    
    return None

def get_player_stats_by_name_from_db(db: Session, player_id: str, match_id: Optional[int]):
    query = db.query(model.CricketersLifetimeStats)
    
    # Apply filters based on match_id
    if match_id:
        # Join with PlayerStats table if match_id is provided
        query = query.join(
            model.PlayerStats,
            model.CricketersLifetimeStats.identifier == model.PlayerStats.player_id
        ).filter(model.PlayerStats.match_id == match_id)
    
    # Join with PlayerBattingStats table
    query = query.join(
        model.PlayerBattingStats,
        model.CricketersLifetimeStats.identifier == model.PlayerBattingStats.player_id
    )
    
    # Join with PlayerBowlingStats table
    query = query.join(
        model.PlayerBowlingStats,
        model.CricketersLifetimeStats.identifier == model.PlayerBowlingStats.player_id
    )
    
    # Apply player_id filter
    query = query.filter(model.CricketersLifetimeStats.identifier == player_id)
    
    # Get the result
    result = query.first()
    
    if result:
        # Extract CricketersLifetimeStats data
        cricketer_data = result
        player_stats = {key: value for key, value in cricketer_data.__dict__.items() if not key.startswith('_')}
        
        # Extract PlayerStats data if match_id is provided
        if match_id:
            player_data = db.query(model.PlayerStats).filter(
                model.PlayerStats.player_id == cricketer_data.identifier,
                model.PlayerStats.match_id == match_id
            ).first()
            if player_data:
                player_dict = {key: value for key, value in player_data.__dict__.items() if not key.startswith('_')}
            else:
                player_dict = {}
        else:
            player_dict = {}
        
        # Extract PlayerBattingStats data
        batting_data = db.query(model.PlayerBattingStats).filter(
            model.PlayerBattingStats.player_id == cricketer_data.identifier
        ).first()
        if batting_data:
            batting_dict = {key: value for key, value in batting_data.__dict__.items() if not key.startswith('_')}
        else:
            batting_dict = {}
        
        # Extract PlayerBowlingStats data
        bowling_data = db.query(model.PlayerBowlingStats).filter(
            model.PlayerBowlingStats.player_id == cricketer_data.identifier
        ).first()
        if bowling_data:
            bowling_dict = {key: value for key, value in bowling_data.__dict__.items() if not key.startswith('_')}
        else:
            bowling_dict = {}
        
        # Transform batting and bowling data
        transformed_batting_data = transform_batting_data(cricketer_data, batting_dict)
        transformed_bowling_data = transform_bowling_data(cricketer_data, bowling_dict)
        
        # Combine and return the data
        return {
            "cricketer_data": {
                "player_id": cricketer_data.identifier,
                "unique_name":cricketer_data.unique_name,
                "full_name":cricketer_data.full_name,
                "player_role":cricketer_data.playing_role,
                "bowling_style":cricketer_data.bowling_style,
                "batting_style":cricketer_data.batting_style,
                },
            "player_data": player_dict,
            "bat": transformed_batting_data,
            "bowl": transformed_bowling_data
        }
    
    return None

# Transformation function for batting data
def transform_batting_data(cricketer_obj, batting_data):
    formats = ["test", "odi", "mdm", "it20", "t20", "odm"]
    batting_mapping = {
        "totalRuns": "total_runs",
        "totalMatches": "total_matches",
        "highestScore": "highest_score",
        "careerStrikeRate": "strike_rate",
        "careerAvg": "avg",
        "last10Avg": "last_10_avg",
        "last10StrikeRate": "last_10_sr",
        "fifty": "fifties",
        "hundred": "hundreds",
        "fours": "fours",
        "sixes": "sixes",
        "stumpings": "stumpings",
        "catches": "catches",
    } 

    static_field_mapping = {
        "adaptability": "adaptability_{fmt}",
        "battingConsistency": "batting_consistency_{fmt}",
        "battingForm": "batting_form_{fmt}",
        "fieldingPerformance": "fielding_performance_{fmt}",
        "match": "all_rounder_performance_{fmt}",
    }
    fields_uppercase = [
        "battingConsistency", "battingForm","match"
    ]
    bat_result = {}
    for fmt in formats:
        # Map batting data
        bat_result[fmt] = {
            new_key: batting_data.get(f"{fmt}_{old_key}", 0.0)
            for new_key, old_key in batting_mapping.items()
        }

        # Add static fields from cricketer_obj
        for new_key, pattern in static_field_mapping.items():
            fmt_value = fmt.upper() if new_key in fields_uppercase else fmt
            cricketer_key = pattern.format(fmt=fmt_value)
            bat_result[fmt][new_key] = getattr(cricketer_obj, cricketer_key, 0.0)

    return bat_result

# Transformation function for bowling data
def transform_bowling_data(cricketer_obj, bowling_data):
    formats = ["test", "odi", "mdm", "it20", "t20", "odm"]
    bowling_mapping = {
        "totalWickets": "total_wickets",
        "totalMatches": "total_matches",
        "bestBowlingInnings": "best_bowling_innings",
        "careerEconomyRate": "economy_rate",
        "careerAvg": "avg",
        "last10Avg": "last_10_avg",
        "last10EconomyRate": "last_10_er",
        "fiveWicketHauls": "five_wickets",
        "tenWicketHauls": "ten_wickets",
        "maidens": "maidens",
    }

    static_field_mapping = {
        "adaptability": "bowling_{fmt}",
        "bowlingConsistency": "bowling_consistency_{fmt}",
        "bowlingForm": "bowling_form_{fmt}",
        "fieldingPerformance": "fielding_performance_{fmt}",
        "match": "all_rounder_performance_{fmt}", 
    }
    fields_uppercase = [
        "bowlingConsistency", "bowlingForm","match"
    ]
    bowl_result = {}
    for fmt in formats:
        # Map bowling data
        bowl_result[fmt] = {
            new_key: bowling_data.get(f"{fmt}_{old_key}", 0.0)
            for new_key, old_key in bowling_mapping.items()
        }

        # Add static fields from cricketer_obj
        for new_key, pattern in static_field_mapping.items():
            fmt_value = fmt.upper() if new_key in fields_uppercase else fmt
            cricketer_key = pattern.format(fmt=fmt_value)
            bowl_result[fmt][new_key] = getattr(cricketer_obj, cricketer_key, 0.0)

    return bowl_result
