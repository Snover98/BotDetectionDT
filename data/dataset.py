import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import Dataset
import data.conn as c
from data.user import User
from data.create_db_to_rnn import is_eng
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
    return users, classes


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
            file = open("../data/users.pickle", 'wb')
            pickle.dump(to_save, file)
            file.close()

        else:
            file = open("../data/users.pickle", 'rb')
            to_save = pickle.load(file)
            self.users, self.labels = to_save
            file.close()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.users[idx], self.labels[idx]


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def get_dataloaders(ds, train_ratio=0.8, batch_size=8):
    train_amount = int(train_ratio * len(ds))
    test_amount = len(ds) - train_amount
    train_set, test_set = data.random_split(ds, (train_amount, test_amount))
    train_dl = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=my_collate)
    test_dl = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=my_collate)
    return train_dl, test_dl


if __name__ == "__main__":
    us_db = UsersDataset()
    count = 0
    for user in us_db.users:
        for tweet in user.tweets:
            count += 1
