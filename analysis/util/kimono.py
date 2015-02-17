import re
from datetime import datetime, timedelta

import requests
import simplejson as json

from statsETL.db.mongolib import *

KIMONO_URL_ROOT = "https://www.kimonolabs.com/api/"
KIMONO_API_KEY = "3GwlWolXpIDwpI5BRyiOqFI2yZT8TCwR"
KIMONO_API_TOKEN = "cr0OJmo4rEyGZ1PAWOlj3lqstCxwzugV"
FANDUEL_DRAFT_PAGE_API = "6a0gv9bw"

def fanDuelNBADraftAPIContent():
    """
    Hits the kimono api called 'fanduel nba draft page',
    which returns data in three collections:

    collection1: available players
    collection2: total salary (get it from somehwere else, THIS WILL CHANGE)
    collection3: roster positions
    """
    content = getKimonoAPIContent(FANDUEL_DRAFT_PAGE_API)
    results = content['results']
    #targeturl = content['targeturl'] fix this
    targeturl = "https://www.fanduel.com/e/Game/11618?tableId=10531011&fromLobby=true"
    game_table_id = parseFanDuelDraftURL(targeturl)
    players = results['collection1']
    salary = results['collection2'][0]['salary']
    salary = int(re.sub('[,$]', '', salary))
    roster_positions = results['collection3']
    roster_positions = [_.values()[0].replace('Add player','').lower() for _ in roster_positions]
    player_names = []
    player_positions = {}
    player_salaries = {}
    player_games = {}
    for player in players:
        raw_name = player['names'].strip()
        gtd = False
        out = False
        if re.search('GTD$', raw_name):
            raw_name = re.sub('GTD$','',raw_name)
            gtd = True
            
        if re.search('O$', raw_name):
            raw_name = re.sub('O$','',raw_name)
            out = True
        if out:
            continue
        if gtd:
            # TODO: DECIDE WHAT TO DO WITH GAME TIME DECISION
            continue
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
    for k,v in pids.iteritems():
        player_dicts_pid[v] = player_dicts[k]
        player_salaries_pid[v] = player_salaries[k]
        player_positions_pid[v] = player_positions[k]
        player_games_pid[v] = player_games[k]

    # TODO: GET THE GAME DATE FROM API
    data = {'budget': salary,
            'roster_positions': roster_positions,
            '_id': game_table_id,
            'players': pids.values(),
            'player_dicts': player_dicts_pid,
            'player_salaries': player_salaries_pid,
            'player_positions': player_positions_pid,
            'game_dates': [datetime(year=2015,month=2,day=19),datetime(year=2015,month=2,day=12)],
            'player_games': player_games_pid} 
    return data

def translatePlayerNames(player_list):
    '''
    TODO: ensure match by team 
    '''
    player_dicts = {}
    for d in player_list:
        name_parts = [x.strip() for x in d.split(' ') if x.strip()]
        query = ''
        for p in name_parts:
            query += '.*%s' % p.replace('.','\.')
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


def changeFanDuelDraftAPISourceURL(new_url):
    response = changeKimonoTargetURL(KIMONO_DRAFT_PAGE_API, new_url)
    return response

def getKimonoAPIContent(kimono_api_id):
    '''
    TODO: NOT GETTING TARGETURL
    '''
    path = "%s?apikey=%s" % (kimono_api_id, KIMONO_API_KEY)
    url = KIMONO_URL_ROOT + path
    response = requests.get(url)
    content = json.loads(response.content)
    return content

def changeKimonoTargetURL(kimono_api_id, new_url):
    post_data = {"apikey" : KIMONO_API_KEY,
                 "targeturl" : new_url}
    path = "%s/update" % (kimono_api_id)
    url = KIMONO_URL_ROOT + path
    response = requests.post(url, post_data)
    return response

def getKimonoTargetURL(kimono_api_id):
    apicontent = getKimonoAPIContent(kimono_api_id)
    targeturl = apicontent['targeturl']
    return targeturl

def startKimonoCrawl(kimono_api_id):
    path = "%s/startcrawl" % kimono_api_id
    post_data = {"apikey" : KIMONO_API_KEY}
    url = KIMONO_URL_ROOT + path
    response = requests.post(url, post_data)
    return response

if __name__ == "__main__":
    data = fanDuelNBADraftAPIContent()
    print data
