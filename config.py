from pymongo import MongoClient
mongodb_uri = 'mongodb+srv://jung:B5QsjWphm3gj5WwS@cluster0.wmmm6.mongodb.net/?retryWrites=true&w=majority'
port = 8000
client = MongoClient(mongodb_uri, port)
db = client["User"]
