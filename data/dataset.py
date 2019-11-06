import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import Dataset
from .conn import connect
from data.user import User
from data.create_db_to_rnn import is_eng
import pickle
import json
import glob
from data.utils import get_tweets_diffs
from data.utils import intensity_indexes
import datetime
from typing import List


def get_users(df: pd.DataFrame):
    ids = df['ID']
    labels = df['Class']
    api = connect()
    users = []
    classes = []
    for id, label in zip(ids, labels):
        try:
            user = User(api, id)
            flag = True
            for tweet in user.tweets:
                if not is_eng(tweet):
                    flag = False
                    break
        except Exception as e:
            flag = False

        if flag:
            users.append(user)
            classes.append(label)
    return users, classes


class UsersDataset(Dataset):

    def __init__(self, csv_file=None, it_flag=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        if csv_file is not None:
            self.users_frame = pd.read_csv(csv_file)
            self.users, self.labels = get_users(self.users_frame)

            to_save = (self.users, self.labels)
            file = open("data/users.pickle", 'wb')
            pickle.dump(to_save, file)
            file.close()

        else:
            file = open("data/users.pickle", 'rb')
            to_save = pickle.load(file)
            self.users, self.labels = to_save
            file.close()

        self.it_flag = it_flag
        if it_flag:
            self.important_topics = get_it()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.it_flag:
            return self.users[idx], self.important_topics[idx], self.labels[idx]
        return self.users[idx], self.labels[idx]


def my_collate(batch):
    if len(batch[0]) == 3:
        user_data, it, target = tuple(zip(*batch))
        target = torch.LongTensor(target)
        return [(user_data, it), target]
    else:
        user_data, target = tuple(zip(*batch))
        target = torch.LongTensor(target)
        return [(user_data,), target]


def second_date_format(file):
    date = file[15:23]
    d = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:8]))
    return d


def save_important_topics():
    # Get the GDELT field names from a helper file
    colnames = pd.read_excel('CSV.header.fieldids.xlsx', sheetname='Sheet1',
                             index_col='Column ID', parse_cols=1)['Field Name']

    us_db = UsersDataset()
    users = us_db.users
    files = glob.glob("db/Gdelttmp/*")
    tweets = [user.tweets for user in users]
    tweets_per_user = [len(user.tweets) for user in users]
    diffs = get_tweets_diffs(tweets)
    intense_indexes = intensity_indexes(diffs, tweets_per_user)

    out = []

    for i, (user, ii) in enumerate(zip(users, intense_indexes)):

        if tweets_per_user[i] == 0:
            out.append([])
            continue
        begin_pos, end_pos = ii
        if end_pos < 0:
            end_pos = 0
        if end_pos - begin_pos > 5:
            end_pos = begin_pos + 5

        begin_date = user.tweets[begin_pos].date + datetime.timedelta(days=-3)
        end_date = user.tweets[end_pos].date
        user_files = [file for file in files if begin_date < second_date_format(file) < end_date]

        if len(user_files) != 0:
            DFlist = []
            for active_file in user_files:
                DFlist.append(pd.read_csv(active_file, sep='\t', header=None, dtype=str,
                                          names=colnames, index_col=['GLOBALEVENTID']))
            df = pd.concat(DFlist)
            df = df.sort_values('GoldsteinScale')
            df = list(df['Actor1Name'].tail(10))
            df = [item for item in df if pd.notna(item)]
        else:
            df = []
        print(df)
        out.append(df)

    with open('listfile.txt', 'w') as filehandle:
        json.dump(out, filehandle)


def get_it():
    with open('data/listfile.txt', 'r') as filehandle:
        important_topics = json.load(filehandle)
    return important_topics


def get_dataloaders(ds: UsersDataset, train_ratio: float, batch_size: int, load_rand_state: bool):
    labels = get_ds_labels_as_np(ds)
    train_indices, test_indices = stratified_train_test_split(labels, train_ratio, load_rand_state)

    train_set = data.Subset(ds, train_indices)
    test_set = data.Subset(ds, test_indices)

    train_dl = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=my_collate)
    test_dl = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=my_collate)
    return train_dl, test_dl


def get_ds_labels_as_np(ds: UsersDataset) -> List[int]:
    dl = data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=my_collate)
    all_labels = torch.cat([batch[-1] for batch in dl])

    return all_labels.numpy()


def stratified_train_test_split(labels: List[int], train_ratio: float, load_rand_state: bool):
    if load_rand_state:
        print("Loading saved random state from rand_state.pickle")
        old_random_state = np.random.get_state()
        state_file = open('rand_state.pickle', 'rb')
        np.random.set_state(pickle.load(state_file))
        state_file.close()
    else:
        print("Saving random state into rand_state.pickle")
        state_file = open('rand_state.pickle', 'wb')
        pickle.dump(np.random.get_state(), state_file)
        state_file.close()

    train_indices, test_indices = train_test_split(np.arange(len(labels)), train_size=train_ratio, stratify=labels)

    if load_rand_state:
        np.random.set_state(old_random_state)

    return train_indices, test_indices


if __name__ == "__main__":
    save_important_topics()
    print(get_it())
