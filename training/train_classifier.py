import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from gensim.models import Word2Vec
from model.classification_model import BotClassifier
from training.training_utils import TorchTrainer, plot_fit
from data.dataset import get_dataloaders, UsersDataset

if __name__ == "__main__":
    num_epochs = 10
    loss_fn = nn.CrossEntropyLoss()

    ds = UsersDataset(it_flag=True)
    train_dl, test_dl = get_dataloaders(ds)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    w2v_model = Word2Vec.load("../checkpoints/word2vec.model")

    clf = BotClassifier(w2v_model, 100, 1024, 128, 1024, use_gdelt=True).to(device)
    optim = Adam(params=clf.parameters(), lr=9e-4)
    trainer = TorchTrainer(clf, loss_fn, optim, device=device)

    fit_res = trainer.fit(train_dl, test_dl, num_epochs, 'clf_checkpoint.model', 3)
    plot_fit(fit_res, legend='First Training')
