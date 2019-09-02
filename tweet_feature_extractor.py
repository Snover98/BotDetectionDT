import torch
from torch import nn
from typing import List
from user import User, Tweet
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import word_training as wt


class TweetFeatureExtractor(nn.Module):
    def __init__(self, word2vec_model, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0):
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

        self.recurrent_extractor = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # at the moment this is without considering additional info about the tweets like the number of mentions, etc...
        # also the structure is arbitrary at the moment
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, inputs: List[List[Tweet]]):
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

        # TASK 1
        # TODO actually use word2vec
        sequences = wt.embed(self.word2vec_model, sum(inputs, []))
        seq_end_lengths_dict = {idx: seq.shape[0] for idx, seq in enumerate(sequences)}
        sorted_indices, sorted_lengths = zip(*sorted(seq_end_lengths_dict.items(), key=lambda x: x[1], reverse=True))
        sorted_indices, sorted_lengths = list(sorted_indices), list(sorted_lengths)
        num_tweets = len(seq_end_lengths_dict)

        # TASK 2
        # DON'T FORGET TO USE PADDING AND PACKING FOR INPUT
        # TODO actually pack the outputs
        padded_seq_batch = pad_sequence(sequences, batch_first=True)
        packed_seq_batch = pack_padded_sequence(padded_seq_batch[sorted_indices], sorted_lengths, batch_first=True)

        # TASK 3
        # DON'T FORGET TO UNDO THE PADDING AND PACKING FROM TASK 3
        recurrent_features, _ = self.recurrent_extractor(packed_seq_batch)
        recurrent_features, _ = pad_packed_sequence(recurrent_features, batch_first=True)

        # TASK 4
        # TODO make sure that this unpacking is correct
        seq_end_indices = [l - 1 for l in sorted_lengths]
        used_recurrent_features = recurrent_features[range(num_tweets), seq_end_indices]
        # also reorder the tweets back
        used_recurrent_features = used_recurrent_features[sorted_indices]

        # TASK 5
        recurrent_features_batch = used_recurrent_features.view(len(inputs), -1, self.hidden_dim)

        # TASK 6
        # TODO add more info for each tweet

        # TASK 7
        recurrent_features_batch = recurrent_features_batch.view(-1, self.hidden_dim)

        # TASK 8
        return self.feature_extractor(recurrent_features_batch).view(len(inputs), -1, self.output_dim)
