import conn
from user import User as US
from tweet_feature_extractor import *
from word_training import train_wtv_on_tweets

api = conn.connect()
user_id = US(api, 3098421349)
print("hi")

model = train_wtv_on_tweets()
print('trained!')
ext = TweetFeatureExtractor(model, 100, 1024, 128)
ext([user_id.tweets])