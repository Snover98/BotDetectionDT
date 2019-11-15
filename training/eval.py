import torch
import torch.nn as nn
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import NamedTuple, List, Tuple
import argparse

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from data.dataset import get_dataloaders, UsersDataset
from training.utils import *


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EvaluationResult(NamedTuple):
    accuracy: float
    f1_score: float
    precision: float
    recall: float


class ModelComparisonResult(NamedTuple):
    accuracies: Tuple[float]
    f1_scores: Tuple[float]
    precisions: Tuple[float]
    recalls: Tuple[float]


class SubrunsModelComparisionResult(NamedTuple):
    LSTM_result: ModelComparisonResult
    TCN_result: ModelComparisonResult
    LSTM_GDELT_result: ModelComparisonResult
    TCN_GDELT_result: ModelComparisonResult


def model_comp_result_from_eval_results(evaluation_results: List[EvaluationResult]):
    return ModelComparisonResult(*tuple(zip(*evaluation_results)))


def plot_model_res_comp_color_map(res: ModelComparisonResult, hyperparam_name: str, hyperparam_vals, subrun_name: str,
                                  model_name: str):
    metrics = ['accuracy', 'f1 score', 'precision', 'recall']
    fig, _ = plot_color_map(np.stack(res), model_name.replace('_', ' ') + " " + subrun_name, hyperparam_name,
                            hyperparam_vals, metrics)
    plt.savefig(f"graphs/{model_name}_{subrun_name.replace(' ', '_')}_metrics.png", bbox_inches="tight")
    plt.close(fig)


def plot_subruns_res_comp_color_map(res: SubrunsModelComparisionResult, hyperparam_name: str, hyperparam_vals,
                                    model_name: str):
    plot_model_res_comp_color_map(res.LSTM_result, hyperparam_name, hyperparam_vals, 'LSTM', model_name)
    plot_model_res_comp_color_map(res.TCN_result, hyperparam_name, hyperparam_vals, 'TCN', model_name)
    plot_model_res_comp_color_map(res.LSTM_GDELT_result, hyperparam_name, hyperparam_vals, 'LSTM with GDELT',
                                  model_name)
    plot_model_res_comp_color_map(res.TCN_GDELT_result, hyperparam_name, hyperparam_vals, 'TCN with GDELT', model_name)


def plot_color_map(mat, title: str, x_label, xticklabels, yticklabels, cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mat, interpolation='nearest', cmap=cmap)
    # We want to show all ticks...
    ax.set(xticks=np.arange(mat.shape[1]),
           yticks=np.arange(mat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=xticklabels, yticklabels=yticklabels,
           title=title,
           xlabel=x_label)

    ax.set_xticks(np.arange(mat.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - .5, minor=True)

    # Loop over data dimensions and create text annotations.
    thresh = mat.mean()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], '.5f'),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


def eval_torch_classifier(model, test_dl, subrun_name: str = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    tot_loss = 0.0

    y_trues, y_preds = [], []

    for batch, labels in test_dl:
        labels = labels.to(device)
        y_hat = model.forward(*batch)

        tot_loss += loss_fn(y_hat, labels).item()

        y_trues.append(labels)
        y_preds.append(y_hat.argmax(dim=1))

    y_true = torch.cat(y_trues).numpy()
    y_pred = torch.cat(y_preds).numpy()

    avg_loss = tot_loss / y_pred.shape[0]

    print(f"The test average loss is:\t{avg_loss}")
    eval_results(y_true, y_pred, subrun_name, verbose=True)


def extract_dataloader_features(trained_extractor: BotClassifier, dl):
    trained_extractor.eval()
    df = pd.DataFrame()

    for batch, labels in dl:
        labels = labels.cpu().numpy()
        ids = [user.id for user in batch[0]]

        features = trained_extractor(*batch).cpu().detach().numpy()
        batch_df = pd.DataFrame(data=features)
        batch_df.loc[:, 'class'] = labels
        batch_df.loc[:, 'user_id'] = ids

        df = df.append(batch_df, ignore_index=True)

    return df


def extract_and_save_features(trained_model: BotClassifier, train_dl, test_dl, subrun_name: str):
    trained_model.classifier = Identity()

    train_df = extract_dataloader_features(trained_model, train_dl)
    test_df = extract_dataloader_features(trained_model, test_dl)

    train_df.to_csv(f"saved_features/{subrun_name}_train.csv")
    test_df.to_csv(f"saved_features/{subrun_name}_test.csv")


def load_train_test_features(subrun_name: str):
    train_df = pd.read_csv(f"saved_features/{subrun_name}_train.csv")
    test_df = pd.read_csv(f"saved_features/{subrun_name}_test.csv")

    return train_df, test_df


def split_df_ids_classes_features(df: pd.DataFrame):
    classes = df['class']
    user_ids = df['user_id']
    features = df.drop('class', axis=1).drop('user_id', axis=1)

    return user_ids, classes, features


def eval_sklearn_model(model, train_df, test_df, subrun_name: str = None):
    train_user_ids, train_classes, train_features = split_df_ids_classes_features(train_df)
    test_user_ids, test_classes, test_features = split_df_ids_classes_features(test_df)

    model.fit(train_features, train_classes)
    test_pred = model.predict(test_features)

    return eval_results(test_classes, test_pred, subrun_name)


def eval_results(y_true, y_pred, subrun_name: str = None, verbose: bool = False):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = accuracy_score(y_true, y_pred)

    if verbose:
        print(f"The test accuracy is:\t{accuracy}")
        print(f"The test f1 score is:\t{f1_score}")
        print(f"The test precision score is:\t{precision}")
        print(f"the test recall score is:\t{recall}")

    if subrun_name is not None:
        confusion_matrix_title = subrun_name.replace('_', ' ') + " Confusion Matrix"
        fig, ax = plot_confusion_matrix(y_true, y_pred, ['Human', 'Bot'], title=confusion_matrix_title)
        plt.savefig(f"graphs/{confusion_matrix_title.replace(' ', '_')}.png")
        plt.close(fig)

    return EvaluationResult(accuracy, f1_score, precision, recall)


def get_init_args(hyperparam_name: str, hyperparam_value, other_init_params: dict):
    init_args = {hyperparam_name: hyperparam_value}
    init_args.update(other_init_params)
    return init_args


def compare_model_by_hyperparam_values(train_df, test_df, ModelClass, other_init_params: dict, hyperparam_name: str,
                                       hyperparam_vals: List):
    evaluation_results = [
        eval_sklearn_model(ModelClass(**get_init_args(hyperparam_name, hyperparam_value, other_init_params)),
                           train_df, test_df)
        for hyperparam_value in hyperparam_vals
    ]

    return model_comp_result_from_eval_results(evaluation_results)


def compare_subruns_by_hyperparam_values(run_name: str, ModelClass, other_init_params: dict, hyperparam_name: str,
                                         hyperparam_vals: List):
    comp_results = []
    for subrun_name in get_all_subrun_names(run_name):
        train_df, test_df = load_train_test_features(subrun_name)
        comp_results.append(
            compare_model_by_hyperparam_values(train_df, test_df, ModelClass, other_init_params, hyperparam_name,
                                               hyperparam_vals)
        )

    return SubrunsModelComparisionResult(*comp_results)


def plot_model_comparison(run_name: str, hyperparam_name: str, hyperparam_vals: List,
                          comp_res: SubrunsModelComparisionResult, fig=None, legend=None):
    if fig is None:
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(32, 20),
                                 sharex='col', sharey='row')
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    subruns = [suffix[len(run_name) + 1:] for suffix in get_all_subrun_names(run_name)]
    metrics = ["accuracies", "f1_scores", "precisions", "recalls"]

    p = itertools.product(metrics, subruns)
    for idx, (metric, subrun) in enumerate(p):
        ax = axes[idx]
        subrun_data = getattr(comp_res, f"{subrun}_result")
        data = getattr(subrun_data, metric)
        h = ax.plot(hyperparam_vals, data, label=legend)

        metric_name = metric[:-1]
        if metric == "accuracies":
            metric_name = 'accuracy'

        if idx // 4 == 0:
            ax.set_title(f"{subrun.replace('_', ' ')}")

        ax.set_xlabel(hyperparam_name)
        ax.set_ylabel(metric_name)
        if legend:
            ax.legend()

    return fig, axes


def plot_similar_models(run_name: str, model_names: List[str], hyperparam_name: str, hyperparam_vals: List,
                        comp_results: List[SubrunsModelComparisionResult]):
    fig = None
    for model_name, comp_res in zip(model_names, comp_results):
        fig, _ = plot_model_comparison(run_name, hyperparam_name, hyperparam_vals, comp_res, fig, model_name)

    return fig


def plot_tsne(df: pd.DataFrame, subrun_name: str):
    tsne = TSNE(n_components=2, verbose=False)

    user_ids, classes, features = split_df_ids_classes_features(df)

    tsne_results = tsne.fit_transform(features)

    fig, ax = plt.subplots()

    label_names = ["Human", "Bot"]
    for idx, color in enumerate(['tab:blue', 'tab:orange']):
        used_tsne_results = tsne_results[classes == idx]
        ax.scatter(used_tsne_results[:, 0], used_tsne_results[:, 1], c=color, alpha=0.3, label=label_names[idx])

    fig.suptitle(f"{subrun_name.replace('_', ' ')} TSNE")
    ax.legend()
    plt.savefig(f"graphs/{subrun_name}_tsne.png", bbox_inches="tight")
    plt.close(fig)


def plot_subrun_tsne(subrun_name: str):
    train_df, test_df = load_train_test_features(subrun_name)
    plot_tsne(train_df, f"{subrun_name}_train")
    plot_tsne(test_df, f"{subrun_name}_test")


def plot_all_tsne():
    run_name = "Final_Training"
    for subrun_name in get_all_subrun_names(run_name):
        plot_subrun_tsne(subrun_name)


def extract_features():
    run_name = "Final_Training"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = UsersDataset(it_flag=True)
    train_dl, test_dl = get_dataloaders(ds, train_ratio=0.8, batch_size=8, load_rand_state=True)

    for use_gdelt, use_TCN in itertools.product([False, True], [False, True]):
        subrun_name = get_subrun_name(run_name, use_gdelt, use_TCN)
        print("Extracting features for " + subrun_name.replace('_', ' '))
        clf = create_model(use_gdelt, use_TCN)
        clf.load_state_dict(torch.load(f"checkpoints/{subrun_name}.model", map_location=device))
        extract_and_save_features(clf, train_dl, test_dl, subrun_name)

    print("Finished extraction")


def evaluate_pytorch_models():
    run_name = "Final_Training"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = UsersDataset(it_flag=True)
    _, test_dl = get_dataloaders(ds, train_ratio=0.8, batch_size=8, load_rand_state=True)

    results = []

    for use_gdelt, use_TCN in itertools.product([False, True], [False, True]):
        subrun_name = get_subrun_name(run_name, use_gdelt, use_TCN)
        print("Evaluating predictions for " + subrun_name.replace('_', ' '))
        clf = create_model(use_gdelt, use_TCN)
        clf.load_state_dict(torch.load(f"checkpoints/{subrun_name}.model", map_location=device))
        results.append(eval_torch_classifier(clf, test_dl, subrun_name))

    results = model_comp_result_from_eval_results(results)
    model_names = ['LSTM', 'TCN', 'LSTM with GDELT', 'TCN with GDELT']

    plot_model_res_comp_color_map(results, "Model", model_names, "Models Evaluation", "Original")


def eval_KNN():
    run_name = "Final_Training"
    print('===========================')
    print("KNN eval")
    print('===========================')
    # KNN with Uniform & distance, hyperparam=K (3, 5, 7, 9, 15, 20, 30, 40, 50)
    K_vals = (3, 5, 7, 9, 15, 20, 30, 40, 50)
    # Uniform
    uniform_results = compare_subruns_by_hyperparam_values(run_name, KNeighborsClassifier, {}, 'n_neighbors', K_vals)
    plot_subruns_res_comp_color_map(uniform_results, 'n_neighbors', K_vals, 'KNN_Uniform')
    # Distance
    distance_results = compare_subruns_by_hyperparam_values(run_name, KNeighborsClassifier, {'weights': 'distance'},
                                                            'n_neighbors', K_vals)
    plot_subruns_res_comp_color_map(distance_results, 'n_neighbors', K_vals, 'KNN_Distance')
    # plot results
    KNN_names = ['Uniform', 'Distance']
    KNN_results = [uniform_results, distance_results]
    fig = plot_similar_models(run_name, KNN_names, 'n_neighbors', K_vals, KNN_results)
    fig.suptitle("KNN")
    plt.savefig(f"graphs/KNN_results.png", bbox_inches="tight")
    plt.close(fig)


def eval_SVM():
    run_name = "Final_Training"

    print('===========================')
    print("SVM eval")
    print('===========================')
    # SVM with kernels (Linear, Poly, Rbf, Sigmoid), hyperparam=C (0.25, 0.5, 1, 5)
    C_vals = (0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0)
    # linear
    linear_results = compare_subruns_by_hyperparam_values(run_name, SVC, {'kernel': 'linear', 'gamma': 'scale'}, 'C',
                                                          C_vals)
    plot_subruns_res_comp_color_map(linear_results, 'C', C_vals, 'SVM_Linear')
    # Poly
    poly_results = compare_subruns_by_hyperparam_values(run_name, SVC, {'kernel': 'poly', 'gamma': 'scale'}, 'C',
                                                        C_vals)
    plot_subruns_res_comp_color_map(poly_results, 'C', C_vals, 'SVM_Poly')
    # Rbf
    rbf_results = compare_subruns_by_hyperparam_values(run_name, SVC, {'gamma': 'scale'}, 'C', C_vals)
    plot_subruns_res_comp_color_map(rbf_results, 'C', C_vals, 'SVM_Rbf')
    # Sigmoid
    sigmoid_results = compare_subruns_by_hyperparam_values(run_name, SVC, {'kernel': 'sigmoid', 'gamma': 'scale'}, 'C',
                                                           C_vals)
    plot_subruns_res_comp_color_map(sigmoid_results, 'C', C_vals, 'SVM_Sigmoid')
    # plot results
    SVM_names = ['Linear', 'Poly', 'Rbf', 'Sigmoid']
    SVM_results = [linear_results, poly_results, rbf_results, sigmoid_results]
    fig = plot_similar_models(run_name, SVM_names, 'C', C_vals, SVM_results)
    fig.suptitle("SVM")
    plt.savefig(f"graphs/SVM_results.png", bbox_inches="tight")
    plt.close(fig)


def eval_trees():
    run_name = "Final_Training"
    print('===========================')
    print("Decision Tree eval")
    print('===========================')
    # Decision Tree with different max features (None, 0.6, log2, auto, 0.8), hyperparam=min_samples_split (2, 5, 10, 30, 50)
    min_samples_vals = (2, 5, 10, 30, 50)
    # None
    none_results = compare_subruns_by_hyperparam_values(run_name, DecisionTreeClassifier, {}, "min_samples_split",
                                                        min_samples_vals)
    plot_subruns_res_comp_color_map(none_results, 'min_samples_split', min_samples_vals, 'Tree_None')
    # 0.6
    point6_results = compare_subruns_by_hyperparam_values(run_name, DecisionTreeClassifier, {'max_features': 0.6},
                                                          "min_samples_split", min_samples_vals)
    plot_subruns_res_comp_color_map(point6_results, 'min_samples_split', min_samples_vals, 'Tree_06')
    # log2
    log2_results = compare_subruns_by_hyperparam_values(run_name, DecisionTreeClassifier, {'max_features': 'log2'},
                                                        "min_samples_split", min_samples_vals)
    plot_subruns_res_comp_color_map(log2_results, 'min_samples_split', min_samples_vals, 'Tree_log2')
    # auto
    auto_results = compare_subruns_by_hyperparam_values(run_name, DecisionTreeClassifier, {'max_features': 'auto'},
                                                        "min_samples_split", min_samples_vals)
    plot_subruns_res_comp_color_map(auto_results, 'min_samples_split', min_samples_vals, 'Tree_auto')
    # 0.8
    point8_results = compare_subruns_by_hyperparam_values(run_name, DecisionTreeClassifier, {'max_features': 0.8},
                                                          "min_samples_split", min_samples_vals)
    plot_subruns_res_comp_color_map(point8_results, 'min_samples_split', min_samples_vals, 'Tree_08')
    # plot results
    tree_names = ['None', '0.6', 'log2', 'auto', '0.8']
    tree_results = [none_results, point6_results, log2_results, auto_results, point8_results]
    fig = plot_similar_models(run_name, tree_names, 'min_samples_split', min_samples_vals, tree_results)
    fig.suptitle("Decision Tree")
    plt.savefig(f"graphs/Trees_results.png", bbox_inches="tight")
    plt.close(fig)


def eval_rand_forest():
    run_name = "Final_Training"

    print('===========================')
    print("Random Forest eval")
    print('===========================')
    # Random Forest with different number of estimators (2, 5, 10, 20, 30, 50, 90) hyperparam=min_samples_split (2, 5, 10, 30, 50)
    min_samples_vals = (2, 5, 10, 30, 50)
    num_estimators_vals = (2, 5, 10, 20, 30, 50, 90)
    random_forest_results = [
        compare_subruns_by_hyperparam_values(run_name, RandomForestClassifier, {'n_estimators': n_estimators},
                                             'min_samples_split', min_samples_vals)
        for n_estimators in num_estimators_vals
    ]
    # print results
    for n_estimators, results in zip(num_estimators_vals, random_forest_results):
        plot_subruns_res_comp_color_map(results, 'min_samples_split', min_samples_vals, f'Rand_Forest_{n_estimators}')

    # plot results
    random_forest_names = [str(val) for val in num_estimators_vals]
    fig = plot_similar_models(run_name, random_forest_names, 'min_samples_split', min_samples_vals,
                              random_forest_results)
    fig.suptitle("Random Forest")
    plt.savefig(f"graphs/Rand_Forest_results.png", bbox_inches="tight")
    plt.close(fig)


def eval_adaboost():
    run_name = "Final_Training"

    print('===========================')
    print("AdaBoost eval")
    print('===========================')
    # AdaBoost with DecisionTrees, hyperparam=number of estimators (10, 50, 100, 200)
    # Using SAMME.R
    num_estimators_vals = (10, 50, 100, 200)
    SAMME_R_results = compare_subruns_by_hyperparam_values(run_name, AdaBoostClassifier, {}, 'n_estimators',
                                                           num_estimators_vals)
    plot_subruns_res_comp_color_map(SAMME_R_results, 'n_estimators', num_estimators_vals, 'AdaBoost_SAMME_R')
    # Using SAMME
    SAMME_results = compare_subruns_by_hyperparam_values(run_name, AdaBoostClassifier, {'algorithm': 'SAMME'},
                                                         'n_estimators', num_estimators_vals)
    plot_subruns_res_comp_color_map(SAMME_results, 'n_estimators', num_estimators_vals, 'AdaBoost_SAMME')
    # plot results
    adaboost_names = ['SAMME.R', 'SAMME']
    adaboost_results = [SAMME_R_results, SAMME_results]
    fig = plot_similar_models(run_name, adaboost_names, 'n_estimators', num_estimators_vals, adaboost_results)
    fig.suptitle("AdaBoost")
    plt.savefig(f"graphs/AdaBoost_results.png", bbox_inches="tight")
    plt.close(fig)


def evaluate_sklearn_models():
    eval_KNN()
    eval_SVM()
    eval_trees()
    eval_rand_forest()
    eval_adaboost()


def evaluate_models():
    evaluate_pytorch_models()
    evaluate_sklearn_models()


def main():
    parser = argparse.ArgumentParser(description='Run the eval program.')
    subparsers = parser.add_subparsers(title='Subcommands',
                                       description='extract_features, evaluate_classifiers, plot_tsne')
    parser.set_defaults(func=lambda: print("Please choose a subcommand, use --help if you are confused"))

    # The subparser for the feature extraction
    extraction_parser = subparsers.add_parser('extract_features',
                                              help='Extract the features of the users using our trained models.')
    extraction_parser.set_defaults(func=extract_features)

    # The subparser for the model evaluation
    eval_parser = subparsers.add_parser('evaluate_classifiers', help='Evaluate all classifiers.')
    eval_parser.set_defaults(func=evaluate_models)

    # The subparser for tsne plotting
    tsne_parser = subparsers.add_parser('plot_tsne', help='Plot tsne for all 4 subruns.')
    tsne_parser.set_defaults(func=plot_all_tsne)

    parser.parse_args().func()


if __name__ == "__main__":
    main()
