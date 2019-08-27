from tweet_feature_extractor import TweetFeatureExtractor
from user import User
import torch
from torch import nn
from typing import List


class BotClassifier(nn.Module):
    def __init__(self, word2vec_model, embedding_dim, rec_hidden_dim, tweet_features_dim, hidden_dim, num_rec_layers=1,
                 rec_dropout=0):
        super().__init__()
        self.tweet_feature_extractor = TweetFeatureExtractor(word2vec_model, embedding_dim, rec_hidden_dim,
                                                             tweet_features_dim, num_layers=num_rec_layers,
                                                             dropout=rec_dropout)

        self.hidden_dim = hidden_dim

        # does not account for the addition of general user data to the tensors
        self.classifier = nn.Sequential(
            nn.Linear(tweet_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax()
        )

    def forward(self, inputs: List[User]):
        """
        TODO:
        1) use the tweet feature extractor on the users
        2) combine the features of each user's tweets into a single feature vector per user
        3) add some more general user data to the batch (for each user)
        4) classify using these features
        """

        # TASK 1
        users_tweets_features = self.tweet_feature_extractor([user.tweets for user in inputs])

        # TASK 2
        # TODO combine each user's tweets' feature vectors into a single feature vector

        # TASK 3
        # TODO add more data about each user

        # TASK 4
        return self.classifier(users_tweets_features)

