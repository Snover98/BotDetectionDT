from torch.utils.data import Dataset
import pandas as pd
import torch
import conn as c
from user import User
from create_db_to_rnn import is_eng
import pickle


def get_users(df: pd.DataFrame):
    ids = df['ID']
    labels = df['Class']
    api = c.connect()
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

    return users, labels


class UsersDataset(Dataset):

    def __init__(self, csv_file=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        if csv_file is not None:
            self.users_frame = pd.read_csv(csv_file)
            self.users, self.labels = get_users(self.users_frame)

            to_save = (self.users, self.labels)
            file = open("users.pickle", 'wb')
            pickle.dump(to_save, file)
            file.close()

        else:
            file = open("users.pickle", 'rb')
            to_save = pickle.load(file)
            self.users, self.labels = to_save
            file.close()

    def __len__(self):
        return len(self.users_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.users[idx], self.labels[idx]


if __name__ == "__main__":
    us_db = UsersDataset()

