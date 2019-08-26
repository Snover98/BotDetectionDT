import conn
from user import User as US

api = conn.connect()
user_id = US(api, 3098421349)
print("hi")
