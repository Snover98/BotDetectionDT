import tweepy
import conn


class Tweet:
    def __init__(self, tweet: tweepy.api):
        self.author = tweet.author
        self.date = tweet.created_at
        self.entities = tweet.entities
        self.favorite_count = tweet.favorite_count
        self.is_quote = tweet.is_quote_status
        self.retweet_count = tweet.retweet
        if hasattr(tweet, 'retweeted_status'):
            self.retweeted_status = Tweet(tweet.retweeted_status)
        else:
            self.retweeted_status = None
        self.text = tweet.full_text


class User:
    def __init__(self, api: tweepy.api, user_id: int):
        time_line = api.user_timeline(user_id, tweet_mode='extended')
        user_d = api.get_user(user_id, tweet_mode='extended')
        self.id = user_id
        self.screen_name = user_d.screen_name
        self.description = user_d.description
        self.followers_count = user_d.followers_count
        self.friends_count = user_d.friends_count
        self.lang = user_d.lang
        self.name = user_d.name
        self.image_url = user_d.profile_image_url_https
        self.status_description = None
        self.tweets = [Tweet(tweet) for tweet in time_line]
