import torch
from torch.utils import data
from torch import nn
from dataset import UsersDataset
from gensim.models import Word2Vec
from classification_model import BotClassifier
from torch.optim.adam import Adam
from training import TorchTrainer, plot_fit

num_epochs = 10
batch_size = 8
optim = Adam(lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

ds = UsersDataset(None)
train_ratio = 0.8
train_amount = int(train_ratio * len(ds))
test_amount = len(ds) - train_amount

train_set, test_set = data.random_split(ds, (train_amount, test_amount))
train_dl = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_dl = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

w2v_model = Word2Vec.load("word2vec.model")
clf = BotClassifier(w2v_model, 100, 1024, 128, 1024)

trainer = TorchTrainer(clf, loss_fn, optim, device='cuda' if torch.cuda.is_available() else 'cpu')

fit_res = trainer.fit(train_dl, test_dl, num_epochs, 'clf_checkpoint.model', 3)
plot_fit(fit_res, legend='First Training')


