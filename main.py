from fastapi import FastAPI, HTTPException, Depends,status, File, UploadFile
from typing import List
from hashing import Hash
from jwttoken import create_access_token
from oauth import get_current_user, get_current_active_user
from models import User
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from config import db

from s import ssd
from modular import play
# s는 cropping 알고리즘, modular는 cluster 알고리즘 파일입니다.
# 서버를 염두에 두고 개발하던 중 코랩에서 최종 개발하는 것으로 결정되었습니다.

import uvicorn
import os

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/uploadfiles")
async def create_upload_files(num : int, current_user:User = Depends(get_current_user),  files: List[UploadFile] = File(...) ):
    UPLOAD_DIRECTORY = "./"
    realname = current_user.username

    if num != current_user.target_num:
        db["users"].update_one({'username': realname}, {'$set': {'target_num': num}})

    for file in files:
        contents = await file.read()
        with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:
            fp.write(contents)

    for file in files:
        if(file.filename[-4:] == ".mp4"):
            if(current_user.filename == "video.mp4"):
                db["users"].update_one({'username': realname}, {'$set': {'filename': file.filename}})

        elif(file.filename[-4:] == ".png"):
            if(current_user.targetname == "default.png"):
                db["users"].update_one({'username': realname}, {'$set': {'targetname': file.filename}})

    return {"filenames": [file.filename for file in files]}

@app.post('/register')
async def create_user(request:User):
	hashed_pass = Hash.bcrypt(request.password)
	user_object = dict(request)
	user_object["password"] = hashed_pass
	user_id = db["users"].insert_one(user_object)
	return {"register":"completed"}

@app.post('/login')
async def login(request:OAuth2PasswordRequestForm = Depends()):
	user = db["users"].find_one({"username":request.username})
	if not user:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail = f'No user found with this {request.username} username')
	if not Hash.verify(user["password"],request.password):
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail = f'Wrong Username or password')
	access_token = create_access_token(data={"sub": user["username"] })
	return {"access_token": access_token, "token_type": "bearer"}

@app.put("/play")
async def algo(current_user:User = Depends(get_current_active_user)):
#algorithm

    list_name = []
    list_name.append(current_user.targetname)
    numb = current_user.target_num

    ssd(list_name)
    cnt = play(numb,list_name)
    print(cnt)
    score = 0
    if cnt < 20:
        score = 20
    elif cnt < 50:
        score = 50
    elif cnt < 80:
        score = 80
    elif cnt < 100:
        score = 100
    elif cnt < 150:
        score = 150
    else:
        score = 200
#update
    realname = current_user.username
    user_score = current_user.score + score
    db["users"].update_one({'username': realname}, {'$set': {'score': user_score}})

    return {cnt : score}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)