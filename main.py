import tweepy as tp

consumer_key = "4r5cXFmBMaOWiy5R5NQATAddP"
secret_key = "GEm9d4MI5Ra2MJYLwSrrGJMiMZHvTgUkP4dbjNfN4kCO2pWvZb"
at = "216640185-G8yc1A9ggUfXyaOomvSFhrgsxJk3uPiftnFuuwBL"
sat = "B55uySDe7AeYOA4KJwqdnVRctakX3RfQddakdxa5bnvlp"

auth = tp.OAuthHandler(consumer_key, secret_key)
auth.set_access_token(at, sat)

api = tp.API(auth)
user_id = 338914454
time_line = api.user_timeline(user_id)
user = api.get_user(user_id)
print("hi")
