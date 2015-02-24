import re
from datetime import datetime, timedelta
import time

import requests
import simplejson as json

from statsETL.db.mongolib import *

KIMONO_URL_ROOT = "https://www.kimonolabs.com/kimonoapis"
KIMONO_URL_DATA = "https://www.kimonolabs.com/api"
KIMONO_API_KEY = "3GwlWolXpIDwpI5BRyiOqFI2yZT8TCwR"
KIMONO_API_TOKEN = "cr0OJmo4rEyGZ1PAWOlj3lqstCxwzugV"
FANDUEL_DRAFT_PAGE_API = "6a0gv9bw"
NBA_ROSTER_URLS_API = "apxs63v4"
NBA_ROSTER_API = "26djfnzm"

def fanDuelNBADraftAPIContent(targeturl):
    """
    Hits the kimono api called 'fanduel nba draft page',
    which returns data in three collections:

    collection1: available players
    collection2: total salary (get it from somehwere else, THIS WILL CHANGE)
    collection3: roster positions

    """
    # TODO: lock from here
    changed = changeKimonoTargetURL(FANDUEL_DRAFT_PAGE_API, targeturl)
    results = crawlAndGetContent(FANDUEL_DRAFT_PAGE_API, crawl=False)
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


def updateNBARosters():
    """
    get all the nba roster urls,
    then crawl for the roster
    """
    results = crawlAndGetContent(NBA_ROSTER_URLS_API)
    if not results:
        return
    results = results['collection1']
    roster_urls = {}
    for info_dict in results:
        data = info_dict['team_roster_urls']
        team_name = data['text']
        team_roster_url = data['href']
        roster_urls[team_name] = team_roster_url
    for name, url in roster_urls.iteritems():
        # find team 
        team_row = team_collection.find_one({"name":str(name).strip()})
        if not team_row:
            raise Exception("Could not find %s" % name)
        changed = changeKimonoTargetURL(NBA_ROSTER_API, url)
        results = crawlAndGetContent(NBA_ROSTER_API)
        if not results:
            continue
        results = results['collection1']
        player_names = []
        for playerdict in results:
            data = playerdict['players']
            pname = data['text']
            player_names.append(pname)
        player_dicts = translatePlayerNames(player_names)
        player_ids = [v["_id"] for k,v in player_dicts.iteritems()]
        team_row['players'] = player_ids
        # save
        team_collection.save(team_row)
        print "Saved %s roster: %s" % (name, player_ids)
        time.sleep(3)


def translatePlayerNames(player_list):
    '''
    TODO: ensure match by team 

    Hardcoded: 
    - Patty -> Patrick
    '''
    translations = {"Patty": "Patrick"}

    player_dicts = {}
    for d in player_list:
        name_parts = [x.strip() for x in d.split(' ') if x.strip()]
        query = ''
        for p in name_parts:
            if p in translations:
                p = translations[p]
            query += '.*%s' % p.replace('.','.*')
        if query:
            query += '.*'
        full_matched = player_collection.find_one({'full_name' : {'$regex' : re.compile(query)}})
        nick_matched = player_collection.find_one({'nickname' : {'$regex' : re.compile(query)}})
        if full_matched or nick_matched:
            match_obj = full_matched or nick_matched
            #print "%s, match: %s" % (d, match_obj)
            player_dicts[d] = match_obj
        else:
            print "%s, no match: %s" % (d, query)
    return player_dicts


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

def getKimonoAPIResults(kimono_api_id):
    path = "/%s?apikey=%s" % (kimono_api_id, KIMONO_API_KEY)
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

def startKimonoCrawl(kimono_api_id):
    path = "/%s/startcrawl" % kimono_api_id
    post_data = {"apikey" : KIMONO_API_KEY}
    url = KIMONO_URL_ROOT + path
    print url
    response = requests.post(url, post_data)
    if response.status_code != 200:
        print response.content
        print response.status_code
        raise Exception(response.content)
    return response

def crawlAndGetContent(kimono_api_id, crawl=True):
    if crawl:
        try:
            crawled = startKimonoCrawl(kimono_api_id)
        except Exception as e:
            print "Failed crawl: %s" % e
            return
    api = getKimonoAPIContent(kimono_api_id)
    while api["lastrunstatus"] != "success":
        time.sleep(1)
        api = getKimonoAPIContent(kimono_api_id)
    results = getKimonoAPIResults(kimono_api_id)
    return results

if __name__ == "__main__":
    #data = fanDuelNBADraftAPIContent()
    data = updateNBARosters()
    print data