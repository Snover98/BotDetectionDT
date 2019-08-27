from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import pandas as pd
import re
from random import randint
import numpy as np
import nltk
import torch
from create_db import get_tweets
from nltk.tokenize import word_tokenize as wt
from typing import List


def w2v_pre_process(string: str, mentions, urls):
    string = re.sub(r"(?<=[\s^])@(?=[\s$])", "at", string)

    # replace mentions with @
    for mention in mentions:
        string = string.replace(f"@{mention}", "@")

    # remove $
    string = string.replace("$", " ")

    # replace the urls with $
    for url in urls:
        string = string.replace(url, "$")

    # remove special chars
    string = re.sub(r"[\']", "", string)
    string = re.sub(r"[^\w@\$]", " ", string)

    tokens = wt(string)
    return tokens


def train(model: Word2Vec, db: pd.DataFrame, num_epoches: int = None):
    word_lists = [w2v_pre_process(row["seq"], row["mentions"], row["urls"]) for _, row in db.iterrows()]
    model.train(word_lists, total_examples=len(word_lists), epochs=num_epoches)
    return model


def train_wtv_on_tweets():
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    df = get_tweets()
    model = train(model, df, 5)
    return model


def embed(model: Word2Vec, tweets: List):
    seq_list = []
    for tweet in tweets:
        urls = tweet.entities["urls"]

        if 'media' in tweet.entities.keys():
            urls += tweet.entities["media"]

        urls = [url['url'] for url in urls]
        mentions = [mention['screen_name'] for mention in tweet.entities["user_mentions"]]

        word_list = w2v_pre_process(tweet.text, mentions, urls)
        seq_list.append(torch.stack([torch.from_numpy(model.wv.word_vec(word)) for word in word_list]))
    return seq_list
