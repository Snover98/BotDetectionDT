from user import User
import pandas as pd
import conn
from langdetect import detect


def is_eng(tweet):
    return detect(tweet.text) == 'en'


def get_users():
    df = pd.read_csv("db/varol-2017.csv")

    ids = df['ID'].astype(int)
    api = conn.connect()
    users = []
    for id in ids:
        user = User(api, id)
        flag = True
        for tweet in user.tweets:
            if not is_eng(tweet):
                flag = False
                break
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


