import requests
import lxml.html as lh
import os.path
import urllib.request
import zipfile
import glob
import operator
import datetime
from data.dataset import UsersDataset
from data.utils import get_tweets_diffs
from data.utils import intensity_indexes
import pickle


def is_between(st, dates):
    if len(st) != 23:
        return False
    st_date = st[:8]
    d = datetime.datetime(int(st_date[:4]), int(st_date[4:6]), int(st_date[6:8]))
    for (begin_date, end_date) in dates:
        if begin_date < d < end_date:
            return True
    return False


if __name__ == "__main__":
    us_db = UsersDataset()
    users = us_db.users
    dates = []
    tweets = [user.tweets for user in users]
    tweets_per_user = [len(user.tweets) for user in users]
    diffs = get_tweets_diffs(tweets)
    intense_indexes = intensity_indexes(diffs, tweets_per_user)

    for i, (user, ii) in enumerate(zip(users, intense_indexes)):

        if tweets_per_user[i] == 0:
            continue
        begin_pos, end_pos = ii
        if end_pos < 0:
            end_pos = 0
        if end_pos - begin_pos > 5:
            end_pos = begin_pos + 5

        begin_date = datetime.timedelta(days=-3) + user.tweets[begin_pos].date
        end_date = user.tweets[end_pos].date
        print(begin_date,end_date)
        dates.append((begin_date, end_date))

    gdelt_base_url = 'http://data.gdeltproject.org/events/'

    # get the list of all the links on the gdelt file page
    page = requests.get(gdelt_base_url + 'index.html')
    doc = lh.fromstring(page.content)
    link_list = doc.xpath("//*/ul/li/a/@href")

    # separate out those links that begin with four digits
    file_list = [x for x in link_list if str.isdigit(x[0:4])]
    file_list = [x for x in file_list if is_between(x, dates)]
    print(len(file_list))
    infilecounter = 0
    outfilecounter = 0

    local_path = 'db/Gdelt'

    for compressed_file in file_list:
        print(f"{compressed_file}")

        # if we dont have the compressed file stored locally, go get it. Keep trying if necessary.
        while not os.path.isfile(local_path + compressed_file):
            print('downloading')
            urllib.request.urlretrieve(url=gdelt_base_url + compressed_file,
                                       filename=local_path + compressed_file)

        # extract the contents of the compressed file to a temporary directory
        print('extracting')
        z = zipfile.ZipFile(file=local_path + compressed_file, mode='r')
        z.extractall(path=local_path + 'tmp/')

        # os.remove(local_path + compressed_file)
        print('done')
