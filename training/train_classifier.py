import sys

sys.path.append('')
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from gensim.models import Word2Vec
import argparse
from model.classification_model import BotClassifier
from training.training_utils import TorchTrainer, plot_fit, display_fit_result
from data.dataset import get_dataloaders, UsersDataset
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the train loop.')
    parser.add_argument('-n', '--run_name', dest='run_name', type=str, required=True,
                        help="""The name of the current run. 
                                Used in the checkpoint and figure names (if these are used).""")
    parser.add_argument('-c', '--use_checkpoint', dest='use_checkpoint', action='store_true', default=False,
                        help="""A flag for using checkpoints in our training.
                            If the flag is not used, checkpoints will not be saved or loaded during the training.""")
    parser.add_argument('-r', '--load_rand_state', dest='load_rand_state', action='store_true', default=False,
                        help="""A flag that's used if we want to use the numpy random state saved in rand_state.pickle 
                            for the train/test split. 
                            If not used, we'll override the rand_state.pickle file 
                            with the default numpy random state used for the split.""")
    parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=0.8,
                        help='The ratio of train samples from the dataset. default=0.8')
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
    parser.add_argument('--effective_history', dest='effective_history', default=60, type=int,
                        help="""The maximum expected sequence length of a tweet after 
                            removing stop words and such things. only relevant when use_TCN=True. default=60.""")
    parser.add_argument('--num_rec_layers', dest='num_rec_layers', default=1, type=int,
                        help="""The number of layers in the LSTM. Only relevant when use_TCN=False. default=1.""")
    parser.add_argument('--rec_dropout', dest='rec_dropout', type=float, nargs='?', const=0.2, default=0.0,
                        help="""The dropout probability used in the temporal extractor.
                            If not used it is 0.0, and if used with no number (i.e using --rec_dropout no value) 
                            it will have a default of 0.2""")
    parser.add_argument('--plot_results', dest='plot_results', action='store_true', default=False,
                        help="A flag that's used if we want to plot the fit results with pyplot.")
    parser.add_argument('--compare_temporal', dest='compare_temporal', action='store_true', default=False,
                        help="""A flag for comparing the temporal extractors.
                            If the flag is not used, we'll use the one specified by the -t flag.""")
    parser.add_argument('--compare_gdelt', dest='compare_gdelt', action='store_true', default=False,
                        help="""A flag for comparing the model with and without gdelt.
                                If the flag is not used, we'll use the one specified by the --use_gdelt flag.""")

    return parser.parse_args()


def main(args):
    if args.plot_results:
        plt.switch_backend('Agg')

    num_epochs = args.num_epochs
    loss_fn = nn.CrossEntropyLoss()

    ds = UsersDataset(it_flag=args.use_gdelt or args.compare_gdelt)
    train_dl, test_dl = get_dataloaders(ds, train_ratio=args.train_ratio, batch_size=args.batch_size,
                                        load_rand_state=args.load_rand_state)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    w2v_model = Word2Vec.load("checkpoints/word2vec.model")

    temporal_options = [False, True] if args.compare_temporal else [args.use_TCN]
    gdelt_options = [False, True] if args.compare_gdelt else [args.use_gdelt]
    fig = None

    for use_gdelt in gdelt_options:
        for use_TCN in temporal_options:
            temporal_ext_name = "TCN" if use_TCN else "LSTM"
            subrun_name = f"{args.run_name}_{temporal_ext_name}"
            if use_gdelt:
                subrun_name += "_GDELT"

            clf = BotClassifier(w2v_model, args.embedding_dim, args.rec_hidden_dim, args.tweet_features_dim,
                                args.hidden_dim, use_gdelt=use_gdelt, use_TCN=use_TCN,
                                effective_history=args.effective_history, num_rec_layers=args.num_rec_layers,
                                rec_dropout=args.rec_dropout).to(device)

            if args.use_SGD:
                optim = SGD(params=clf.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            else:
                optim = Adam(params=clf.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            trainer = TorchTrainer(clf, loss_fn, optim, device=device)

            checkpoint_file = None
            if args.use_checkpoint:
                checkpoint_file = f"{subrun_name}.model"

            print("================================================================================")
            print("===============================|STARTED TRAINING|===============================")
            print("================================================================================")
            print(f'Training with extractor {temporal_ext_name} and use_gdelt={use_gdelt}:')
            fit_res = trainer.fit(train_dl, test_dl, num_epochs, checkpoints=checkpoint_file,
                                  early_stopping=args.early_stopping)
            print("================================================================================")
            print('===================================|FINISHED|===================================')
            print("================================================================================")
            print('')
            display_fit_result(fit_res)

            if args.plot_results:
                fig, _ = plot_fit(fit_res, fig=fig, legend=subrun_name.replace('_', ' '))

    fig.suptitle(args.run_name)
    plt.savefig(f"graphs/{args.run_name}.png")


if __name__ == "__main__":
    main(parse_arguments())
