from statsETL.db.mongolib import *
from statsETL.util.crawler import *
import pandas as pd
from collections import defaultdict

games = nba_games_collection.find()
#games = nba_games_collection.find({"_id":"0021300384"})


player_games = defaultdict(int)
team_games = defaultdict(int)

years_back=[2007,2008,2009,2010,2011,2012,2013,2014,2015]

for y in years_back[::-1]:
    games = nba_games_collection.find({"season": str(y)})
    for g in games:
        date = g['date']
        season = g['season']
        team_stats = pd.read_json(g['TeamStats'])
        player_stats= pd.read_json(g['PlayerStats'])
        usage_stats = pd.read_json(g["PlayerTrack"])
        scoring_stats = pd.read_json(g["sqlPlayersScoring"])
        '''
        if len(team_stats.index) == 0:
            print "team stats empty"
            continue
        '''
        if len(player_stats.index) == 0 or len(usage_stats.index) == 0 or len(scoring_stats.index)==0:
            print "player stats empty"
            continue
        if 'NET_RATING' not in player_stats:
            print "no advanced stats"
            continue
        team_games[season] += 1
        print team_games
        #print team_games

        '''
        matched = True
        for i, p_df in player_stats.iterrows():
            pid = int(p_df['PLAYER_ID'])
            gid = str(p_df['GAME_ID']).zfill(10)
            query = {"game_id": gid, "player_id": pid}
            shotcharts = list(shot_chart_collection.find(query))
            shots = list(shot_collection.find(query))
            if len(shotcharts) == 0:
                matched = False
                break
            
            elif len(shotcharts) == 0:
                matched = False
                break

            elif len(shots) != len(shotcharts):
                matched = False
                break
            else:
                pass
            
        if matched:
            player_games[season] += 1
            print player_games
        '''