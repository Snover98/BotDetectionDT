from pywikibot.data import api
import pywikibot
import pprint

PYWIKIBOT_NO_USER_CONFIG = 1


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


def get_info(items: list):
    site, repo, token = wiki_connect()
    output = []
    for item in items:
        item_output = []
        wikidataEntries = getItems(site, item)
        for wdEntry in wikidataEntries["search"]:
            item_output.append(wdEntry['description'])
        output.append(item_output)
    return output


if __name__ == '__main__':
    # prettyPrint(getItem(site, wdEntry["id"], token))
    print(get_info(["israel", "trump"]))
