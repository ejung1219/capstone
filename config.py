import os
from pymongo import MongoClient
from dotenv import load_dotenv
# github에는 올리지 않은 .env 파일에서 mongodb uri와
# jwt token 발행 시 사용하는 SECRET KEY를 가져옵니다.
# 기본 세팅을 위한 모듈입니다.

load_dotenv()

class Settings:
    mongodb_uri : str = os.getenv("mongodb_uri")
    SERCET_KEY : str = os.getenv("SECRET_KEY")

settings = Settings()

mongodb_uri = settings.mongodb_uri
port = 8000
client = MongoClient(mongodb_uri, port)
db = client["User"]
