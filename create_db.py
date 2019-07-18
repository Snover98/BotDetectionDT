import sqlite3
from sqlite3 import Error
from user import User
import conn as c


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
            print(query)
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

    create_table = """CREATE TABLE IF NOT EXISTS tweets ( user_id integer,date_time text NOT NULL,
                            favorite_count integer , is_quote integer, tweet_text text NOT NULL,
                             FOREIGN KEY(user_id) REFERENCES users(id));"""
    ex.execute(create_table)

    create_table = """CREATE TABLE IF NOT EXISTS mentions ( tweet_id integer, mention text NOT NULL);"""
    ex.execute(create_table)

    create_table = """CREATE TABLE IF NOT EXISTS urls ( tweet_id integer, url text NOT NULL);"""
    ex.execute(create_table)


def insert_data(ex):
    pass


def reset_database(ex):
    query = "DROP TABLE users;"
    ex.execute(query)
    query = "DROP TABLE tweets;"
    ex.execute(query)
    query = "DROP TABLE mentions;"
    ex.execute(query)
    query = "DROP TABLE urls;"
    ex.execute(query)


if __name__ == '__main__':
    conn = create_connection("pythonsqlite.db")
    ex = conn.cursor()

    create_tabels(ex)
    insert_data(ex)

    conn.commit()
    ex.close()
    conn.close()
