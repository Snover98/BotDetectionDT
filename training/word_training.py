from gensim.models import Word2Vec
import re
import torch
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


def train(model: Word2Vec, word_lists, num_epoches: int = None):
    model.build_vocab(word_lists, progress_per=10000)
    model.train(word_lists, total_examples=len(word_lists), epochs=num_epoches)
    return model


def get_text(users):
    word_lists = []
    tweets = sum([user.tweets for user in users], [])
    for tweet in tweets:
        urls = tweet.entities["urls"]

        if 'media' in tweet.entities.keys():
            urls += tweet.entities["media"]

        urls = [url['url'] for url in urls]
        mentions = [mention['screen_name'] for mention in tweet.entities["user_mentions"]]

        word_lists.append(w2v_pre_process(tweet.text, mentions, urls))
    return word_lists


def train_wtv_on_tweets(users, num_epochs=5):
    model = Word2Vec(size=100, window=5, min_count=1, workers=4)
    model = train(model, get_text(users), num_epochs)
    return model


def embed(model: Word2Vec, tweets: List, device):
    seq_list = []
    for tweet in tweets:
        urls = tweet.entities["urls"]

        if 'media' in tweet.entities.keys():
            urls += tweet.entities["media"]

        urls = [url['url'] for url in urls]
        mentions = [mention['screen_name'] for mention in tweet.entities["user_mentions"]]

        word_list = w2v_pre_process(tweet.text, mentions, urls)

        seq_list.append(torch.stack([torch.from_numpy(model.wv.word_vec(word)).to(device) for word in word_list]))


    return seq_list
