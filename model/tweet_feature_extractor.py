import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple
from data.user import Tweet
from training.word_training import embed
from .utils import get_TCN_params_from_effective_history
from TCN.tcn import TemporalConvNet


class TweetFeatureExtractor(nn.Module):
    def __init__(self, word2vec_model, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2,
                 use_gdelt=False,
                 use_TCN=False, effective_history=91):
        """

        :param word2vec_model: the actual model that would embed the tweets
        :param embedding_dim:
        :param hidden_dim:
        :param num_layers:
        :param dropout:
        """
        super().__init__()

        self.word2vec_model = word2vec_model
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gdelt = use_gdelt

        self.use_TCN = use_TCN

        if use_TCN:
            num_levels, kernel_size = get_TCN_params_from_effective_history(effective_history)
            num_channels = [hidden_dim] * num_levels
            self.temporal_extractor = TemporalConvNet(embedding_dim, num_channels, kernel_size, dropout)
        else:
            self.temporal_extractor = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # at the moment this is without considering additional info about the tweets like the number of mentions, etc...
        # also the structure is arbitrary at the moment
        self.num_handmade_features = 4
        self.feature_extractor = nn.Linear(hidden_dim + self.num_handmade_features, output_dim)

    @staticmethod
    def sorted_seq_by_len(sequences) -> Tuple[List[int], List[int]]:
        seq_end_lengths_dict = {idx: seq.shape[0] for idx, seq in enumerate(sequences)}
        sorted_indices, sorted_lengths = zip(*sorted(seq_end_lengths_dict.items(), key=lambda x: x[1], reverse=True))
        return list(sorted_indices), list(sorted_lengths)

    def forward(self, inputs: List[List[Tweet]], tweets_per_user: List[int]):
        """
        TODO:
        1) use word2vec to create as sequence of vectors for each tweet
            (i.e each tweet is a sequence of words)
        2) create a batch out of all of the sequences (num_users*tweets_per_user, max_seq_len, embedding_dim)
            (make sure to remember which tweets belong to which users)
        3) feed the batch into the recurrent feature extractor (num_users*tweets_per_user, max_seq_len, hidden_dim)
        4) use only the last output (or the last few outputs) of each sequence.
        5) create a tensor for each user made out of the tensors that came out of their tweets (concat or something)
        6) add some other relevant data about each tweet to the tensors (like post time and stuff like that)
        7) create a batch from those tensors (num_users, hidden_dim*tweets_per_user)
        8) feed these these tensors into the linear feature extractor and return it's output

        """
        handmade_features = []

        # TASK 1
        device = next(self.parameters()).device
        sequences = embed(self.word2vec_model, sum(inputs, []), device)

        sorted_indices, sorted_lengths = self.sorted_seq_by_len(sequences)
        num_tweets = len(sorted_indices)

        # TASK 2
        # DON'T FORGET TO USE PADDING AND PACKING FOR INPUT
        padded_seq_batch = pad_sequence(sequences, batch_first=True)
        if self.use_TCN:
            padded_seq_batch = torch.stack([m.t() for m in padded_seq_batch[sorted_indices]])
        else:
            packed_seq_batch = pack_padded_sequence(padded_seq_batch[sorted_indices], sorted_lengths, batch_first=True)
        # TASK 3
        # DON'T FORGET TO UNDO THE PADDING AND PACKING FROM TASK 3
        if self.use_TCN:
            recurrent_features = torch.stack([m.t() for m in self.temporal_extractor(padded_seq_batch)])
        else:
            recurrent_features, _ = self.temporal_extractor(packed_seq_batch)
            recurrent_features, _ = pad_packed_sequence(recurrent_features, batch_first=True)

        # TASK 4
        seq_end_indices = [l - 1 for l in sorted_lengths]
        used_recurrent_features = recurrent_features[range(num_tweets), seq_end_indices]
        # also reorder the tweets back
        used_recurrent_features = used_recurrent_features[sorted_indices]

        # ADD HANDMADE FEATURES
        # favorite count
        handmade_features.append(
            torch.Tensor([tweet.favorite_count for tweet in sum(inputs, [])]).to(device).unsqueeze(1))
        # is the tweet a quote?
        handmade_features.append(torch.Tensor([tweet.is_quote for tweet in sum(inputs, [])]).to(device).unsqueeze(1))
        # whether or not there is a retweeted status
        handmade_features.append(
            torch.Tensor([tweet.retweeted_status is None for tweet in sum(inputs, [])]).to(device).unsqueeze(1))
        # the number of entities in the tweet
        handmade_features.append(
            torch.Tensor([sum([len(entity) for entity in tweet.entities.values()]) for tweet in sum(inputs, [])]).to(
                device).unsqueeze(1))

        features_dim = self.hidden_dim + self.num_handmade_features
        used_recurrent_features = torch.cat((used_recurrent_features, *handmade_features), dim=1)

        # TASK 5
        used_recurrent_features = list(torch.split(used_recurrent_features, tweets_per_user))

        for i, urf in enumerate(used_recurrent_features):
            dim0 = urf.shape[0]
            if dim0 != 100:
                used_recurrent_features[i] = torch.cat(
                    [urf, torch.zeros(100 - dim0, features_dim, device=urf.device)], 0)

        used_recurrent_features = torch.cat(used_recurrent_features)

        recurrent_features_batch = used_recurrent_features.view(len(inputs), -1, features_dim)

        # TASK 7
        recurrent_features_batch = recurrent_features_batch.view(-1, features_dim)

        # TASK 8
        return self.feature_extractor(recurrent_features_batch).view(len(inputs), -1, self.output_dim)
