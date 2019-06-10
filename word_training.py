from gensim.models import Word2Vec
import pandas as pd
import re
from random import randint


def train(model: Word2Vec, db: pd.DataFrame, num_batch: int, num_epoches: int):
    max_batch = db["batch_idx"].max()

    if num_batch > max_batch:
        num_batch = max_batch

    batches = [randint(0, max_batch) for _ in range(num_batch)]
    batch = db["batch_idx" in batches]

    seqs = batch["seq"]
    word_list = []
    for seq in seqs:
        word_list.append(re.sub("[^\w]", " ", seq).split())
        
    model.train(word_list, total_examples=len(word_list), epochs=num_epoches)
