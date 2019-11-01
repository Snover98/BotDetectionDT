import sys

sys.path.append('../')
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from gensim.models import Word2Vec
import argparse
from model.classification_model import BotClassifier
from training.training_utils import TorchTrainer, plot_fit
from data.dataset import get_dataloaders, UsersDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the train loop.')
    parser.add_argument('-c', '--checkpoint_file', dest='checkpoint_file', type=str, default=None,
                        help="""The name of the checkpoint file we'd like to use.
                            If the flag is not used, checkpoints will not be saved or loaded during the training.""")
    parser.add_argument('-e', '--epochs', dest='num_epochs', default=25, type=int,
                        help='The number of epochs used in the training.')
    parser.add_argument('-t', '--use_TCN', dest='use_TCN', action='store_true', default=False,
                        help="""A flag that's used if we want to use a TCN as the temporal extractor.
                            If not used, an LSTM will be the temporal extractor.""")
    parser.add_argument('--use_gdelt', dest='use_gdelt', action='store_true', default=False,
                        help="A flag that's used if we want to use gdelt in the model.")
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=4,
                        help='The batch size for the training.')
    parser.add_argument('--early_stopping', dest='early_stopping', type=int, default=None,
                        help="""Number of epochs without improvement until early stopping.
                            By default there is no early stopping.""")
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=9e-4,
                        help='The learning rate for the classifier.')
    parser.add_argument('-w', '--weight_decay', dest='weight_decay', type=float, default=0,
                        help='The L2 regularisation coefficient. default=0.')
    parser.add_argument('--use_SGD', dest='use_SGD', action='store_true', default=False,
                        help="A flag that's used if we want to use SGD as our optimiser instead of ADAM.")
    parser.add_argument('--embedding_dim', dest='embedding_dim', default=100, type=int,
                        help="""The embedding dimension of the wordToVec model. default=100.
                            WARNING: if you change it from 100, you must retrain the wordToVec model.""")
    parser.add_argument('--rec_hidden_dim', dest='rec_hidden_dim', default=1024, type=int,
                        help="""The hidden dimension of the temporal extractor. default=1024.""")
    parser.add_argument('--tweet_features_dim', dest='tweet_features_dim', default=128, type=int,
                        help="""The output dimension of the temporal extractor. default=128.""")
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=1024, type=int,
                        help="""The hidden dimension of the classifier. default=1024.""")
    parser.add_argument('--effective_history', dest='effective_history', default=91, type=int,
                        help="""The maximum expected sequence length of a tweet after 
                            removing stop words and such things. only relevant when use_TCN=True. default=91.""")
    parser.add_argument('--num_rec_layers', dest='num_rec_layers', default=1, type=int,
                        help="""The number of layers in the LSTM. Only relevant when use_TCN=False. default=1.""")
    parser.add_argument('--rec_dropout', dest='rec_dropout', type=float, nargs='?', const=0.2, default=0.0,
                        help="""The dropout probability used in the temporal extractor.
                            If not used it is 0.0, and if used with no number (i.e using --rec_dropout no value) 
                            it will have a default of 0.2""")
    parser.add_argument('--plot_results', dest='plot_results', action='store_true', default=False,
                        help="A flag that's used if we want to plot the fit results with pyplot.")

    return parser.parse_args()


def main(args):
    num_epochs = args.num_epochs
    loss_fn = nn.CrossEntropyLoss()

    ds = UsersDataset(it_flag=True)
    train_dl, test_dl = get_dataloaders(ds, batch_size=args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    w2v_model = Word2Vec.load("../checkpoints/word2vec.model")

    clf = BotClassifier(w2v_model, args.embedding_dim, args.rec_hidden_dim, args.tweet_features_dim, args.hidden_dim,
                        use_gdelt=args.use_gdelt, use_TCN=args.use_TCN, effective_history=args.effective_history,
                        num_rec_layers=args.num_rec_layers, rec_dropout=args.rec_dropout).to(device)

    if args.use_SGD:
        optim = SGD(params=clf.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optim = Adam(params=clf.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    trainer = TorchTrainer(clf, loss_fn, optim, device=device)

    fit_res = trainer.fit(train_dl, test_dl, num_epochs, checkpoints=args.checkpoint_file,
                          early_stopping=args.early_stopping)
    print("Fit result:")
    print(fit_res)

    if args.plot_results:
        plot_fit(fit_res, legend='First Training')


if __name__ == "__main__":
    main(parse_arguments())
