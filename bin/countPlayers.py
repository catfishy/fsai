from statsETL.db.mongolib import *
from statsETL.util.crawler import *
from statsETL.bball.NBAcrawler import *
import pandas as pd
from collections import defaultdict

by_year = defaultdict(int)
players = nba_players_collection.find()
data = defaultdict(int)
for p in players:
    data['total'] += 1
    found = False
    if len(p.get('BR_POSITION',[])) > 0:
        data['br_pos'] += 1
        found = True
    else:
        crawlNBAPlayer(p['_id'])
    if len(p['POSITION']) > 0:
        data['pos'] += 1
        found = True
    if not found:
        data['missing'] += 1
    #print data
