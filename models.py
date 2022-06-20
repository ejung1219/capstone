from pydantic import BaseModel
from typing import Optional

# model들을 정의하는 모듈입니다.

class User(BaseModel):
    username: str
    score : int = 0
    password: str
    filename : str = "video.mp4"
    targetname : str = "default.png"
    target_num : int = 3

class Login(BaseModel):
	username: str
	password: str
class Token(BaseModel):
    access_token: str
    token_type: str
class TokenData(BaseModel):
    username: Optional[str] = None
