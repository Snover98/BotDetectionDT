from data.user import User
import pandas as pd
from .conn import connect
from langdetect import detect


def is_eng(tweet):
    return detect(tweet.text) == 'en'


def get_users():
    df = pd.read_csv("db/varol-2017.csv")
    ids = df['ID']
    api = connect()
    users = []
    bad_user = []
    for id in ids:
        try:
            user = User(api, id)
            flag = True
            for tweet in user.tweets:
                if not is_eng(tweet):
                    flag = False
                    break
        except Exception as e:
            bad_user.append(id)
            flag = False

        if flag:
            users.append(user)
    return users


def create_db_for_rnn():
    db = pd.DataFrame(columns=["seq_idx", "seq", "mentions", "urls"])
    users = get_users()

    for idx, user in enumerate(users):
        for tweet in user.tweets:
            db.append({"seq_idx": idx, "seq": tweet.text, "mentions": tweet.entities["user_mentions"],
                       "urls": tweet.entities["urls"] + tweet.entities["media"]})
    return db


