import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

class Settings:
    mongodb_uri : str = os.getenv("mongodb_uri")
    SERCET_KEY : str = os.getenv("SECRET_KEY")

settings = Settings()

mongodb_uri = settings.mongodb_uri
port = 8000
client = MongoClient(mongodb_uri, port)
db = client["User"]
