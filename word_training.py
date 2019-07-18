from gensim.models import Word2Vec
import pandas as pd
import re
from random import randint
import numpy as np
import nltk
from nltk.tokenize import word_tokenize as wt


def w2v_pre_process(string: str, mentions, urls):
    string = re.sub(r"(?<=[\s\A])@(?=[\s\Z])", "at", string)

    # replace mentions with @
    for mention in mentions:
        screen_name = mention["screen_name"]
        string = string.replace(f"@{screen_name}", "@")

    # remove $
    string = string.replace("$", " ")

    # replace the urls with $
    for url in urls:
        url_str = url["url"]
        string = string.replace(url_str, "$")

    # remove special chars
    string = re.sub(r"[^\w@\$]", " ", string)

    tokens = wt(string)
    return tokens


def train(model: Word2Vec, db: pd.DataFrame, batch_size: int, num_epoches: int = None):
    word_lists = [w2v_pre_process(row["seq"], row["mentions"], row["urls"]) for _, row in db.iterrows()]
    model.train(word_lists, total_examples=len(word_lists), epochs=num_epoches)
    return model
