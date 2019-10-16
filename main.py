import torch
from torch import nn
from dataset import UsersDataset
from gensim.models import Word2Vec
from classification_model import BotClassifier
from torch.optim.adam import Adam
from training import TorchTrainer, plot_fit
from dataset import get_dataloaders
from word_training import train_wtv_on_tweets

if __name__ == "__main__":
    num_epochs = 10
    loss_fn = nn.CrossEntropyLoss()

    ds = UsersDataset(None)
    train_dl, test_dl = get_dataloaders(ds)

    w2v_model = Word2Vec.load("word2vec.model")

    clf = BotClassifier(w2v_model, 100, 1024, 128, 1024)
    optim = Adam(params=clf.parameters(), lr=1e-3)
    trainer = TorchTrainer(clf, loss_fn, optim, device='cuda' if torch.cuda.is_available() else 'cpu')

    fit_res = trainer.fit(train_dl, test_dl, num_epochs, 'clf_checkpoint.model', 3)
    plot_fit(fit_res, legend='First Training')

