import itertools
from gensim.models import Word2Vec
from model.classification_model import BotClassifier
import torch


def get_subrun_name(run_name: str, use_gdelt: bool, use_TCN: bool):
    temporal_ext_name = "TCN" if use_TCN else "LSTM"
    subrun_name = f"{run_name}_{temporal_ext_name}"
    if use_gdelt:
        subrun_name += "_GDELT"

    return subrun_name


def get_all_subrun_names(run_name: str):
    p = itertools.product([False, True], [False, True])
    return tuple([get_subrun_name(run_name, use_gdelt, use_TCN) for use_gdelt, use_TCN in p])


def create_model(use_gdelt: bool, use_TCN: bool):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    w2v_model = Word2Vec.load("checkpoints/word2vec.model")

    rec_hidden_dim = 1024
    if use_TCN:
        rec_hidden_dim = 256

    return BotClassifier(w2v_model, 100, rec_hidden_dim, 256, 1024, use_gdelt=use_gdelt, use_TCN=use_TCN,
                         effective_history=60, num_rec_layers=1, rec_dropout=0.0).to(device)
