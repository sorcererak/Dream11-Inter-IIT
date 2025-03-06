from sqlalchemy.orm import Session
from app import model


def get_all_teams_from_db(db: Session):
    return db.query(model.Team).all()

def get_teams_by_name_from_db(db: Session, team_nameA: str, team_nameB: str):
    return db.query(model.Team).filter(model.Team.name.in_([team_nameA, team_nameB])).all()

def get_team_by_id_from_db(db: Session, team_id: int):
    return db.query(model.Team).filter(model.Team.id == team_id).first()

def get_team_info_by_name_from_db(db: Session, team_name: str):
    return db.query(model.Team).filter(model.Team.name == team_name).first()