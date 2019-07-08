import conn
from user import User as US

api = conn.connect()
user_id = US(api, 4066646536)
print("hi")
