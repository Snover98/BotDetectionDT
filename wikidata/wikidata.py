from pywikibot.data import api
import pywikibot
import pprint
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from training.word_training import w2v_pre_process

PYWIKIBOT_NO_USER_CONFIG = 1


def similar(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def getItems(site, itemtitle):
    params = {'action': 'wbsearchentities', 'format': 'json', 'language': 'en', 'type': 'item', 'search': itemtitle}
    request = api.Request(site=site, **params)
    return request.submit()


def getItem(site, wdItem, token):
    request = api.Request(site=site,
                          action='wbgetentities',
                          format='json',
                          ids=wdItem)
    return request.submit()


def prettyPrint(variable):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(variable)


def wiki_connect():
    site = pywikibot.Site("wikidata", "wikidata")
    repo = site.data_repository()
    token = repo.token(pywikibot.Page(repo, 'Main Page'), 'edit')
    return site, repo, token


def get_info(site, repo, token, items: list):
    output = []
    for item in items:
        item_output = ""
        wikidataEntries = getItems(site, item)

        for wdEntry in wikidataEntries["search"]:
            if 'description' not in wdEntry.keys():
                continue
            if wdEntry['description'] != 'Wikimedia disambiguation page':
                item_output = f"{item_output} {wdEntry['description']}"

        output.append(item_output)
    return output


def calculate_similarity_wikidata(tweet_lists, topics, intense_indexes):
    similarity = []
    site, repo, token = wiki_connect()
    for user_tweets, user_topics, intense_index in zip(tweet_lists, topics, intense_indexes):
        all_user_words = []
        sim = 0

        start_pos, end_pos = intense_index
        if end_pos - start_pos > 5:
            end_pos = start_pos + 5

        for tweet in user_tweets[start_pos: end_pos]:
            urls = tweet.entities["urls"]

            if 'media' in tweet.entities.keys():
                urls += tweet.entities["media"]

            urls = [url['url'] for url in urls]
            mentions = [mention['screen_name'] for mention in tweet.entities["user_mentions"]]

            word_list = w2v_pre_process(tweet.text, mentions, urls)
            word_list = [word for word in word_list if word not in stopwords.words('english')]
            all_user_words += word_list

        wikidata_expension = get_info(site, repo, token, all_user_words) + all_user_words

        for topic in user_topics:
            for word in wikidata_expension:
                sim += similar(topic, word)

        similarity.append(sim)

    return similarity


if __name__ == '__main__':
    # prettyPrint(getItem(site, wdEntry["id"], token))
    res = 0
    # Initializing strings
    test_string1 = 'UNITED STATES'
    print(get_info(["trump"])[0])
    for sentence in get_info(["trump"])[0]:
        res += similar(test_string1, sentence)

    # printing the result
    print("The similarity between 2 strings is : " + str(res))
