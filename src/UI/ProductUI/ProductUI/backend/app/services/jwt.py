from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt import PyJWKClient
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        jwt.decode(token, "your_secret_key", algorithms=["HS256"])
        return True
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    