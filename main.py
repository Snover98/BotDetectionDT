import conn
from user import User as US
from tweet_feature_extractor import *
from word_training import train_wtv_on_tweets
from gensim.models import Word2Vec

api = conn.connect()
user_id = US(api, 3098421349)
print("hi")

# model = train_wtv_on_tweets()
# model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
print('trained!')
ext = TweetFeatureExtractor(model, 100, 1024, 128)
print(ext([user_id.tweets]))