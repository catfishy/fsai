import re
from datetime import datetime, timedelta
import time
from collections import defaultdict

import requests
import simplejson as json
from pymongo.helpers import DuplicateKeyError

from statsETL.db.mongolib import *
from statsETL.bball.NBAcrawler import updateNBARosterURLS, translatePlayerNames, translateTeamNames

KIMONO_URL_ROOT = "https://www.kimonolabs.com/kimonoapis"
KIMONO_URL_DATA = "https://www.kimonolabs.com/api"
KIMONO_API_KEY = "3GwlWolXpIDwpI5BRyiOqFI2yZT8TCwR"
KIMONO_API_TOKEN = "cr0OJmo4rEyGZ1PAWOlj3lqstCxwzugV"
FANDUEL_DRAFT_PAGE_API = "6a0gv9bw"
NBA_ROSTER_URLS_API = "apxs63v4"
NBA_ROSTER_API = "26djfnzm"
TEAMSTAT_API = "3qmi47kk"

def updateNBARosters():
    """
    get all the nba roster urls,
    then crawl for the roster
    """
    url = "http://espn.go.com/nba/players"
    results = crawlAndGetContent(NBA_ROSTER_URLS_API, crawl=False)
    if not results:
        return
    results = results['collection1']
    roster_urls = {}
    for info_dict in results:
        data = info_dict['team_roster_urls']
        team_name = data['text']
        team_roster_url = data['href']
        roster_urls[team_name] = team_roster_url
    updateNBARosterURLS(roster_urls)

def getESPNTeamStats():
    x = datetime.now()
    today = datetime(day=x.day,month=x.month,year=x.year)

    results = crawlAndGetContent(TEAMSTAT_API, crawl=False, with_url=True)
    results = results['collection1']

    by_url = defaultdict(list)

    results[1].pop('url')
    key_translate = {k: v['text'] for k,v in results[1].iteritems()}
    team_names = []
    for row in results[2:]:
        url = row.pop('url')
        translated = {}
        valid_row = False
        for k,v in row.iteritems():
            try:
                newval = float(v['text'])
                valid_row = True # valid row if holding some floats
            except Exception as e:
                newval = v['text']
            translated[key_translate[k]] = newval
        if not valid_row:
            continue
        team_names.append(translated['TEAM'])
        by_url[url].append(translated)
    team_translate = translateTeamNames(team_names)

    for k,values in by_url.items():
        to_return = {}
        for v in values:
            name = v.pop('TEAM')
            to_return[team_translate[name]] = v
        
        # derive date from url
        if '/year/' in k:
            # spoof end of season
            year = int(k.split('/')[-1])
            date = datetime(day=1,month=7,year=year)
        else:
            date = today

        # save to database
        to_save = {"time" : date, "stats": to_return}

        try:
            nba_conn.saveDocument(espn_stat_collection, to_save)
        except DuplicateKeyError as e:
            print "%s espn team stats already saved" % date

def fanDuelNBADraftAPIContent(targeturl, crawl=True):
    """
    Hits the kimono api called 'fanduel nba draft page',
    which returns data in three collections:

    collection1: available players
    collection2: total salary (get it from somehwere else, THIS WILL CHANGE)
    collection3: roster positions

    """
    # TODO: lock from here
    changed = changeKimonoTargetURL(FANDUEL_DRAFT_PAGE_API, targeturl)
    results = crawlAndGetContent(FANDUEL_DRAFT_PAGE_API, crawl=crawl)
    # to here

    game_table_id = parseFanDuelDraftURL(targeturl)
    players = results['collection1']
    salary = results['collection2'][0]['salary']
    salary = int(re.sub('[,$]', '', salary))
    roster_positions = results['collection3']
    roster_positions = [_.values()[0].replace('Add player','').lower() for _ in roster_positions]
    player_names = []
    invalid_names = []
    gtds = []
    player_positions = {}
    player_salaries = {}
    player_games = {}
    for player in players:
        raw_name = player['names'].strip()
        gtd = False
        out = False
        na = False
        ir = False
        if re.search('GTD$', raw_name):
            raw_name = re.sub('GTD$','',raw_name)
            gtd = True
        if re.search('O$', raw_name):
            raw_name = re.sub('O$','',raw_name)
            out = True
        if re.search('NA$', raw_name):
            raw_name = re.sub('NA$','',raw_name)
            na = True
        if re.search('IR$', raw_name):
            raw_name = re.sub('IR$','',raw_name)
            ir = True

        # TODO: BE SMARTER ABOUT THIS (GIVE BENEFIT OF DOUBT)
        if out or ir or na:
            invalid_names.append(raw_name)
        if gtd:
            gtds.append(raw_name)
        player_salary = int(re.sub('[,$]', '', player['salaries']))
        player_game = player['game']
        player_position = player['positions'].lower()
        player_names.append(raw_name)
        player_positions[raw_name] = player_position
        player_salaries[raw_name] = player_salary
        player_games[raw_name] = player_game

    # TRANSLATE NAMES AND MATCH PLAYER IDS
    player_dicts = translatePlayerNames(player_names)
    pids = {k:v['_id'] for k,v in player_dicts.iteritems()}
    player_salaries_pid = {}
    player_positions_pid = {}
    player_games_pid = {}
    player_dicts_pid = {}
    invalid_pids = []
    gtd_pids = []
    for k,v in pids.iteritems():
        player_dicts_pid[v] = player_dicts[k]
        player_salaries_pid[v] = player_salaries[k]
        player_positions_pid[v] = player_positions[k]
        player_games_pid[v] = player_games[k]
        if k in invalid_names:
            invalid_pids.append(v)
        if k in gtds:
            gtd_pids.append(v)

    data = {'budget': salary,
            'roster_positions': roster_positions,
            '_id': game_table_id,
            'players': pids.values(),
            'player_dicts': player_dicts_pid,
            'player_salaries': player_salaries_pid,
            'player_positions': player_positions_pid,
            'player_games': player_games_pid,
            'invalids': invalid_pids,
            'gtds': gtd_pids
            }
    return data


def parseFanDuelDraftURL(url):
    '''
    url = "https://www.fanduel.com/e/Game/11618?tableId=10531011&fromLobby=true"
    '''
    last_part = url.split('/')[-1]
    id_part = last_part.split('&')[0]
    return id_part

def createFanDuelDraftURL(game_table_id):
    new_url = "https://www.fanduel.com/e/Game/%s" % game_table_id
    return new_url

def getKimonoAPIContent(kimono_api_id):
    path = "/%s?apikey=%s" % (kimono_api_id, KIMONO_API_KEY)
    url = KIMONO_URL_ROOT + path
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(response.content)
    content = json.loads(response.content)
    return content

def getKimonoAPIResults(kimono_api_id, with_url=False):
    path = "/%s?apikey=%s" % (kimono_api_id, KIMONO_API_KEY)
    if with_url:
        path += "&kimwithurl=1"
    url = KIMONO_URL_DATA + path
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(response.content)
    content = json.loads(response.content)
    return content['results']

def changeKimonoTargetURL(kimono_api_id, new_url):
    post_data = {"apikey" : KIMONO_API_KEY,
                 "targeturl" : new_url}
    path = "/%s/update" % (kimono_api_id)
    url = KIMONO_URL_ROOT + path
    print url
    response = requests.post(url, post_data)
    if response.status_code != 200:
        raise Exception(response.content)
    return response

def setKimonoTargetURLS(kimono_api_id, urls):
    post_data = {"apikey" : KIMONO_API_KEY,
                 "urls" : urls}
    path = "/%s/update" % (kimono_api_id)
    url = KIMONO_URL_ROOT + path
    print url
    response = requests.post(url, post_data)
    if response.status_code != 200:
        raise Exception(response.content)
    return response

def startKimonoCrawl(kimono_api_id):
    path = "/%s/startcrawl" % kimono_api_id
    post_data = {"apikey" : KIMONO_API_KEY}
    url = KIMONO_URL_ROOT + path
    print url
    print post_data
    response = requests.post(url, post_data)
    if response.status_code != 200:
        print response.content
        print response.status_code
        raise Exception(response.content)
    return response

def crawlAndGetContent(kimono_api_id, crawl=False, retries=3, with_url=False):
    print "Hitting %s, crawl: %s" % (kimono_api_id, crawl)
    if crawl:
        crawled = None
        for retry in range(retries):
            try:
                crawled = startKimonoCrawl(kimono_api_id)
                print crawled
                break
            except Exception as e:
                print "Failed crawl attempt : %s" % e
        if crawled is None:
            raise Exception("Failed after %s retries" % retries)
    api = getKimonoAPIContent(kimono_api_id)
    while api["lastrunstatus"] != "success":
        time.sleep(1)
        api = getKimonoAPIContent(kimono_api_id)
    results = getKimonoAPIResults(kimono_api_id, with_url=with_url)
    return results

if __name__ == "__main__":
    #data = fanDuelNBADraftAPIContent()
    getESPNTeamStats()
    #getDepthChart()
    updateNBARosters()

