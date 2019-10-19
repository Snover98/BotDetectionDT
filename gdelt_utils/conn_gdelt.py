from google.cloud import bigquery
from data.dataset import UsersDataset
import datetime
from data.utils import intensity_indexes


def date_format(date):
    date = str(date).replace('-', '').split(' ')[0]
    return date


def get_topics_in_dates(tweets, diffs, tweets_per_user):
    client = bigquery.Client.from_service_account_json('../gdelt_utils/botdetection-9ce670130b13.json')
    important_entities = []
    intense_indexes = intensity_indexes(diffs, tweets_per_user)
    for i, (user_tweets, intense_index) in enumerate(zip(tweets, intense_indexes)):
        important_entities.append([])
        start_pos, end_pos = intense_index
        start_tweet_date, end_tweet_date = user_tweets[start_pos].date, user_tweets[end_pos].date

        tweet_date = date_format(end_tweet_date)
        begin_date = date_format(start_tweet_date + datetime.timedelta(days=-7))
        QUERY = (
            f"SELECT Actor1Name from `gdelt-bq.gdeltv2.events` where SQLDATE<={tweet_date} and SQLDATE>={begin_date} ORDER BY GoldsteinScale DESC limit 10;")
        query_job = client.query(QUERY)  # API request
        rows = query_job.result()  # Waits for query to finish

        for row in rows:
            if row[0] is not None and row[0] not in important_entities[i]:
                important_entities[i].append(row[0])

    return important_entities, intense_indexes


# TODO EXPAND WORDS WITH WIKIDATA
# TODO CHECK REFERENCES IN TWEETS

if __name__ == "__main__":
    db = UsersDataset()
    users = db.users[:8]
    ie = get_topics_in_dates(users)
    for i in ie:
        print(i)
