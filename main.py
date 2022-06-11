from fastapi import FastAPI, HTTPException, Depends, Request,status, File, UploadFile
from typing import List
from hashing import Hash
from jwttoken import create_access_token
from oauth import get_current_user, get_current_active_user
from models import User
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from config import db
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

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/files/")
async def create_files(current_user:User = Depends(get_current_user),files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}

@app.post("/uploadfiles")
async def create_upload_files(current_user:User = Depends(get_current_user),files: List[UploadFile] = File(...)):
    UPLOAD_DIRECTORY = "./"
    for file in files:
        contents = await file.read()
        with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:
            fp.write(contents)
        print(file.filename)
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

@app.put("/update")
async def update_student(current_user:User = Depends(get_current_active_user)):
    realname = current_user.username
    user_score = current_user.score
    db["users"].update_one({'username': realname}, {'$set': {'score': user_score + 1}})

    return {"you get your cut!"}

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)