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

        num_tweets_per_user = 100

        self.tweets_combiner = nn.Sequential(
            nn.BatchNorm1d(num_tweets_per_user * tweet_features_dim),
            nn.ReLU(),
            nn.Linear(num_tweets_per_user * tweet_features_dim, tweet_features_dim)
            )

        # does not account for the addition of general user data to the tensors
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(tweet_features_dim),
            nn.ReLU(),
            nn.Linear(tweet_features_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
        users_tweets_features: torch.Tensor = self.tweet_feature_extractor([user.tweets for user in inputs])

        # TASK 2
        users_tweets_features = self.tweets_combiner(users_tweets_features.view(len(inputs), -1))

        # TASK 3
        # TODO add more data about each user

        # TASK 4
        return self.classifier(users_tweets_features)
