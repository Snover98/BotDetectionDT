import conn
from user import User as US
from tweet_feature_extractor import *
from word_training import train_wtv_on_tweets
from gensim.models import Word2Vec
from classification_model import BotClassifier

api = conn.connect()
user1 = US(api, 3098421349)
user2 = US(api, 12052952)
print("hi")

# model = train_wtv_on_tweets()
# model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
print('trained!')
# ext = TweetFeatureExtractor(model, 100, 1024, 128)
# print(ext([user1.tweets, user2.tweets]).shape)
clf = BotClassifier(model, 100, 1024, 128, 1024)
out = clf([user1, user2])
print(out)
print(out.shape)



