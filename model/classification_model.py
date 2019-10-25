from model.tweet_feature_extractor import TweetFeatureExtractor
from data.user import User
import torch
from torch import nn
from typing import List
from wikidata.wikidata import calculate_similarity_wikidata


class BotClassifier(nn.Module):
    def __init__(self, word2vec_model, embedding_dim, rec_hidden_dim, tweet_features_dim, hidden_dim, use_gdelt=False,
                 num_rec_layers=1,
                 rec_dropout=0):
        super().__init__()

        self.use_gdelt = use_gdelt

        self.tweet_feature_extractor = TweetFeatureExtractor(word2vec_model, embedding_dim, rec_hidden_dim,
                                                             tweet_features_dim, num_layers=num_rec_layers,
                                                             dropout=rec_dropout, use_gdelt=self.use_gdelt)

        self.hidden_dim = hidden_dim

        num_tweets_per_user = 100

        self.tweets_combiner = nn.Sequential(
            nn.BatchNorm1d(num_tweets_per_user * tweet_features_dim),
            nn.ReLU(),
            nn.Linear(num_tweets_per_user * tweet_features_dim, tweet_features_dim)
        )

        # account for the addition of general user data to the tensors
        if self.use_gdelt:
            tweet_features_dim += 1

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(tweet_features_dim),
            nn.ReLU(),
            nn.Linear(tweet_features_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax()
        )

    def forward(self, inputs: List[User], important_topics):
        """
        TODO:
        1) use the tweet feature extractor on the users
        2) combine the features of each user's tweets into a single feature vector per user
        3) add some more general user data to the batch (for each user)
        4) classify using these features
        """

        # TASK 1
        tweet_lists = [user.tweets for user in inputs]

        users_tweets_features, intense_indexes = self.tweet_feature_extractor(
            tweet_lists,
            [len(user.tweets) for user in inputs])

        # TASK 2
        users_tweets_features = self.tweets_combiner(users_tweets_features.view(len(inputs), -1))

        # TASK 3
        if self.use_gdelt:
            sims = calculate_similarity_wikidata(tweet_lists, important_topics, intense_indexes)
            sims = torch.Tensor(sims).unsqueeze(1)
            sims /= torch.max(sims)
            users_tweets_features = torch.cat([users_tweets_features, sims], 1)

        # TASK 4
        return self.classifier(users_tweets_features)
