from passlib.context import CryptContext
#hashing을 위한 모듈입니다.

pwd_cxt = CryptContext(schemes =["bcrypt"],deprecated="auto")

class Hash():
	def bcrypt(password:str):
		return pwd_cxt.hash(password)
	def verify(hashed,normal):
		return pwd_cxt.verify(normal,hashed)
