import torch
import torch.nn as nn
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from model.classification_model import BotClassifier


def f_score(confusion, beta=1.0, return_precision_recall=False):
    '''
        computes the general F score based on the given confusion matrix.

        confusion: a n_class X n_class matrix where c[i,j]=number of class i predictions
        where made that according to the ground truth should have been class j

        beta: an averging coeeficient berween precision and recall
    '''
    # precision is number of true class predictions / total class prediction
    # recall is number of true class predictions / number of class in gt
    tp = confusion.diagonal()
    should_be_positive = confusion.sum(0)
    total_positive_predicted = confusion.sum(1)

    class_precision = 100 * (tp / (1e-8 + total_positive_predicted))
    class_recall = 100 * (tp / (1e-8 + should_be_positive))

    score = (1 + beta ** 2) * class_precision * class_recall

    if return_precision_recall:
        return score / (1e-8 + class_recall + (beta ** 2) * class_precision), class_precision, class_recall
    return score / (1e-8 + class_recall + (beta ** 2) * class_precision)


def eval_torch_classifier(model, test_dl, loss_fn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    confusion_mat = torch.zeros(2, 2)
    avg_acc = 0
    num_batches = len(test_dl.batch_sampler)
    avg_loss = 0

    for batch, labels in test_dl:
        batch, labels = batch.to(device), labels.to(device)
        y_hat = model.forward(*batch)

        avg_loss += loss_fn(y_hat, labels)

        avg_acc += ((y_hat.argmax(dim=1) == labels).sum().item()) / len(batch)

        for predict, ground_truth in zip(y_hat, labels):
            confusion_mat[predict, ground_truth] += 1

    f1_score, precision, recall = f_score(confusion_mat, 1, True)
    avg_acc = (avg_acc / num_batches) * 100
    avg_loss = avg_loss / num_batches

    print(f"the avarage accuracy per batch is: {avg_acc}%")
    print(f"the avarage loss per batch is: {avg_loss}")
    print(f"the f_score with beta=1: {f1_score}")
    print(f"the precision per class is: {precision}")
    print(f"the recall per class is: {recall}")

    plt.imshow(confusion_mat.numpy())
    plt.colorbar()
    plt.show()


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def save_featuers_extracted(trained_model: BotClassifier, train_dl, test_dl, train_file, test_file):
    trained_model.classifier = Identity()
    trained_model.eval()
    train_row_list = []
    for batch, labels in train_dl:
        extracted_featuers = trained_model.forward(batch)
        users, _ = batch
        for i, (user, label) in enumerate(zip(users, labels)):
            train_row_list.append({'user_id': user.id, 'class': label, 'featuers_extracted': extracted_featuers[i]})

    train_df = pd.DataFrame(columns=['user_id', 'class', 'featuers_extracted'], data=train_row_list)

    test_row_list = []
    for batch, labels in test_dl:
        extracted_featuers = trained_model.forward(batch)
        users, _ = batch
        for i, (user, label) in enumerate(zip(users, labels)):
            test_row_list.append({'user_id': user.id, 'class': label, 'featuers_extracted': extracted_featuers[i]})

    test_df = pd.DataFrame(columns=['user_id', 'class', 'featuers_extracted'], data=test_row_list)

    train_df.to_csv(train_file)
    test_df.to_csv(test_file)


def eval_sklearn_model(model, train_df, test_df):
    train_featuers = []
    train_gt = [row['class'] for index, row in train_df.iterrows()]

    for index, row in train_df.iterrows():
        train_featuers.append(row['featuers_extracted'])

    model = model.fit(train_featuers, train_gt)

    test_featuers = []
    for index, row in test_df.iterrows():
        test_featuers.append(row['featuers_extracted'])

    pred = model.predict(test_featuers)
    test_gt = [row['class'] for index, row in test_df.iterrows()]

    confusion_matrix = sklearn.metrics.confusion_matrix(test_gt, pred)

    accuracy = 100 * ((pred == test_gt).sum().item() / len(test_df))
    f1_score, precision, recall = f_score(confusion_matrix, 1, True)

    print(f"the avarage accuracy per batch is: {accuracy}%")
    print(f"the f_score with beta=1: {f1_score}")
    print(f"the precision per class is: {precision}")
    print(f"the recall per class is: {recall}")

    plt.imshow(confusion_matrix.numpy())
    plt.colorbar()
    plt.show()
