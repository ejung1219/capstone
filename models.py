from pydantic import BaseModel
from typing import Optional
#
class User(BaseModel):
    username: str
    score : int = 0
    password: str
    filename : str = "video.mp4"
    targetname : str = "target.png"
    target_num : int = 3
class Login(BaseModel):
	username: str
	password: str
class Token(BaseModel):
    access_token: str
    token_type: str
class TokenData(BaseModel):
    username: Optional[str] = None