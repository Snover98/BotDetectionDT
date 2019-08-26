import conn
from user import User as US

api = conn.connect()
user_id = US(api, 256597786)
print("hi")
