import sqlite3
from sqlite3 import Error
from user import User
from create_db_to_rnn import get_users
import pandas as pd


def init_create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        conn.close()


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None


def insert_user(conn, user: User):
    try:
        ex = conn.cursor()
        query = f"""INSERT INTO users (id,screen_name,description,followers_count,friends_count,lang,user_name,img_url)
         VALUES({user.id},'{user.screen_name}','{user.description}',{user.followers_count},{user.friends_count},'{user.lang}'
         ,'{user.name}','{user.image_url}'); """
        ex.execute(query)

        for tweet in user.tweets:

            is_q = 0
            if tweet.is_quote:
                is_q = 1

            query = f"""INSERT INTO tweets(user_id,date_time,favorite_count,is_quote,tweet_text)
            VALUES({user.id},'{tweet.date}',{tweet.favorite_count},{is_q},'{tweet.text.replace("'","")}');"""
            ex.execute(query)

            query = f"""SELECT * FROM tweets;"""
            ex.execute(query)
            result = ex.fetchall()
            result = len(result)

            for mention in tweet.entities["user_mentions"]:
                query = f"""INSERT INTO mentions(tweet_id,mention) VALUES({result},'{mention["screen_name"]}');"""
                ex.execute(query)

            for url in tweet.entities["urls"]:
                query = f"""INSERT INTO urls(tweet_id,url) VALUES({result},'{url["url"]}');"""
                ex.execute(query)

    except Error as e:
        print(e)


def create_tabels(ex):
    # create the tabels
    create_table = """CREATE TABLE IF NOT EXISTS users ( id integer PRIMARY KEY,screen_name text NOT NULL ,description text
                            ,followers_count integer,friends_count integer ,lang text Not NULL ,user_name text NOT NULL, img_url text);"""
    ex.execute(create_table)

    create_table = """CREATE TABLE IF NOT EXISTS tweets ( tweet_id integer PRIMARY KEY AUTOINCREMENT ,user_id integer,date_time text NOT NULL,
                            favorite_count integer , is_quote integer, tweet_text text NOT NULL,
                             FOREIGN KEY(user_id) REFERENCES users(id));"""
    ex.execute(create_table)

    create_table = """CREATE TABLE IF NOT EXISTS mentions ( tweet_id integer, mention text NOT NULL);"""
    ex.execute(create_table)

    create_table = """CREATE TABLE IF NOT EXISTS urls ( tweet_id integer, url text NOT NULL);"""
    ex.execute(create_table)


def insert_data(ex):
    try:
        users = get_users()
        for user in users:
            insert_user(ex, user)
    except Exception as e:
        print("hhhhh")


def reset_database(ex):
    query = "DROP TABLE users;"
    ex.execute(query)
    query = "DROP TABLE tweets;"
    ex.execute(query)
    query = "DROP TABLE mentions;"
    ex.execute(query)
    query = "DROP TABLE urls;"
    ex.execute(query)
    exit()


def show_database(ex):
    query = "SELECT * FROM users;"
    ex.execute(query)
    rows = ex.fetchall()
    print(len(rows))

    query = "SELECT * FROM tweets;"
    ex.execute(query)
    rows = ex.fetchall()
    print(len(rows))

    query = "SELECT * FROM mentions;"
    ex.execute(query)
    rows = ex.fetchall()
    print(len(rows))

    query = "SELECT * FROM urls;"
    ex.execute(query)
    rows = ex.fetchall()
    print(len(rows))
    exit()


def get_tweets():
    df = pd.DataFrame(columns=["seq_idx", "seq", "mentions", "urls"])

    conn = create_connection("pythonsqlite.db")
    ex = conn.cursor()

    query = "SELECT * FROM tweets;"
    ex.execute(query)
    tweets = ex.fetchall()

    query = "SELECT * FROM mentions;"
    ex.execute(query)
    mentions = ex.fetchall()

    query = "SELECT * FROM urls;"
    ex.execute(query)
    urls = ex.fetchall()

    for tweet in tweets:
        tweet_id = tweet[0]
        tweet_text = tweet[1]
        mentions_spesi = [mention[1] for mention in mentions if mention[0] == tweet_id]
        urls_spesi = [url[1] for url in urls if url[0] == tweet_id]
        df.append({"seq_idx": tweet_id, "seq": tweet_text, "mentions": mentions_spesi, "urls": urls_spesi},ignore_index=True)

    return df


if __name__ == '__main__':
    get_tweets()
