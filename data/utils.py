import torch
import numpy as np


def get_tweets_diffs(inputs):
    diffs = []
    for i, user in enumerate(inputs):
        diffs.append([0])
        if len(user) > 1:
            former_date = user[0].date
            for tweet in user[1:]:
                diff = former_date - tweet.date
                diffs[i].append(diff.seconds)
                former_date = tweet.date
        diffs[i] = torch.Tensor(diffs[i])
        if diffs[i].shape[0] < 100:
            diffs[i] = torch.cat([diffs[i], torch.zeros(100 - diffs[i].shape[0])])
    return diffs


def get_tweets_avg_diffs(inputs):
    diffs = []
    for user in inputs:
        if len(user) > 1:
            diffs.append(
                torch.Tensor([(tweet2.date - tweet1.date).seconds for tweet1, tweet2 in zip(user, user[1:])]).mean())
        else:
            diffs.append(torch.Tensor([0.0]))
    return torch.cat(diffs)


def get_max_indexes(diffs):
    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = np.where(np.diff(np.hstack(([False], diffs == 0, [False]))))[0].reshape(-1, 2)

    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    max_len = np.diff(idx_pairs, axis=1).max()
    start_pos = idx_pairs[np.diff(idx_pairs, axis=1).argmax(), 0]
    end_pos = start_pos + max_len - 1
    return start_pos, end_pos


def intensity_indexes(diffs, tweets_per_user):
    max_indexes = []
    for diff, num_tweets in zip(diffs, tweets_per_user):
        real_diff = diff[:num_tweets]
        if 0 in real_diff:
            max_indexes.append(get_max_indexes(real_diff))
        else:
            max_indexes.append((0, num_tweets - 1))
    return max_indexes
