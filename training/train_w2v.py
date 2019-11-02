from training.word_training import train_wtv_on_tweets
from data.dataset import UsersDataset

if __name__ == "__main__":
    ds = UsersDataset(None)
    model = train_wtv_on_tweets(ds.users, 20)
    model.save("checkpoints/word2vec.model")
