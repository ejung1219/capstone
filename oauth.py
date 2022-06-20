from fastapi import Depends,HTTPException, status
from jwttoken import verify_token
from jose import jwt, JWTError
from jwttoken import SECRET_KEY, ALGORITHM
from fastapi.security import OAuth2PasswordBearer
from models import User, TokenData
from config import db

# user 정보를 확인하기 위한 모듈입니다.
# current user 정보를 가져올 수 있습니다.

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class UserInDB(User):
    password: str

def get_user(db, username: str):
    user_dict = db.find_one({"username": username})
    return UserInDB(**user_dict)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db['users'], username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    return current_user