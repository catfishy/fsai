from statsETL.db.mongolib import *
from statsETL.util.crawler import *
from statsETL.bball.NBAcrawler import *
import pandas as pd
from collections import defaultdict

by_year = defaultdict(int)
players = nba_players_collection.find()
data = defaultdict(int)

valid_pos = set(['SG', 'PG', 'SF', 'PF', 'C'])
for p in players:
    data['total'] += 1
    found = False
    br_pos = p.get('BR_POSITION',[])
    invalid_pos = set(br_pos) - valid_pos
    if len(invalid_pos) > 0:
        valids = []
        for pos in br_pos:
            if pos in valid_pos:
                valids.append(pos)
            else:
                if pos == 'GUARD':
                    valids += ['SG', 'PG']
                elif pos == 'PF/C':
                    valids += ['PF', 'C']
                elif pos == 'C/SF':
                    valids += ['SF', 'C']
        # update
        print "%s -> %s" % (br_pos, valids)
        nba_conn.updateDocument(nba_players_collection, p["_id"], {'BR_POSITION': list(set(valids))}, upsert=True)
    elif len(br_pos) > 0:
        data['br_pos'] += 1
        found = True
    else:
        pass
        #crawlNBAPlayer(p['_id'])
    if len(p['POSITION']) > 0:
        data['pos'] += 1
        found = True
    if not found:
        data['missing'] += 1
    #print data
