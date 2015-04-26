"""
Version 1:

- Find out what the upcoming games are
- Find out which players are likely to play which positions by team roster, player positions, and previous minutes played
- For the stats that matter (get points), calculate trajectory for each player in last 15 games
- ? do projections somehow (incorporating matchup context)

"""
from datetime import datetime, timedelta
from collections import defaultdict
import copy
import itertools
import traceback

import numpy as np

from statsETL.db.mongolib import *


def findAllTrainingGames(pid=None, limit=None, time=None, min_count=None):
    # find all training games
    query = {}
    if pid:
        query["player_id"] = pid
    if time:
        query["game_time"] = {"$lt" : time}
    print "Training game query: %s" % query
    playergames = player_game_collection.find(query)

    args = []
    for game in playergames:
        if game['MP'] == 0.0:
            continue
        # create args
        new_arg = {}
        new_arg['player_id'] = str(game['player_id'])
        new_arg['team_id'] = str(game['player_team'])
        new_arg['target_game'] = str(game['game_id'])
        args.append(new_arg)
        # limit
        if limit and len(valid_games) == limit:
            break

    # check above min
    if min_count and len(valid_games) < min_count:
        raise Exception("Couldn't find more than %s games for %s" % (min_count, pid))
    
    return args


class NBAFeatureExtractor(object):

    '''
    TODO: REWRITE OPP PLAYER STARTER/BACKUP SELECTION TO TAKE PLAYER POSITION PERCENTAGES INTO ACCOUNT
    TODO: NORMALIZE GAME NORM ADVANCED STATS BY NUMBER OF GAMES IN CRAWLER
    '''


    # TEAM STATS
    TEAM_ESPN_STATS = ['ESPN_AST', 'ESPN_TS%', 'ESPN_REBR']
    TEAM_STATS = ['Pace', 'ORtg', 'DRtg','NetRtg', 'eFG%', 'ORB%', 'DRB%','TOV%', 
                  'FT/FGA', 'T', 'T_net', 'opp_eFG%', 'opp_TOV%', 'opp_FT/FGA']
    # DRB% = 100 - (opponent's ORB%)
    # DRtg = opponent's ORtg
    # NetRtg = ORtg - DRtg
    # T_net = (own T) - (opp T)
    # reflected stats have opp_ appended

    # PLAYER STATS
    PLAYER_STATS = ["STL%", "ORB%", "BLK%", "FT%", "3PAr", "FG%", 
                    "TOV%", "AST%", "eFG%", "FTr", "+/-", "USG%", "DRB%", "TS%", "MP", "DRtg", "ORtg", "TRB%", 
                    "3P%", "PTS/FGA", "FT_per100poss", "3P_per100poss", "TOV_per100poss", "FG_per100poss", "3PA_per100poss", 
                    "DRB_per100poss", "AST_per100poss", "PTS_per100poss", "FGA_per100poss", 
                    "STL_per100poss", "TRB_per100poss", "FTA_per100poss", "BLK_per100poss", "ORB_per100poss"]
    # PTS/FGA = PTS / FGA
    OPP_PLAYER_STATS = ['DRtg', 'USG%', 'TOV%', 'BLK%', 'STL%', 'TRB%', 'ORB%', 
                        'DRB%', 'FTr', 'TS%', 'eFG%', 'ORtg', '3PAr']
    ADVANCED_STATS = ['PER', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP',
                      'pct_pg', 'pct_sg', 'pct_sf', 'pct_pf', 'pct_c', 'plus_minus_on', 'plus_minus_net',
                      'tov_bad_pass', 'tov_lost_ball', 'tov_other',
                      'fouls_shooting', 'fouls_blocking', 'fouls_offensive', 'fouls_take',
                      'and1s', 'astd_pts', 'fga_blkd', 'drawn_shooting',
                      'fg2_pct','fg_pct_00_03', 'fg_pct_03_10', 'fg_pct_10_16', 'fg_pct_16_XX', 'fg3_pct',
                      'fg2a_pct_fga','pct_fga_00_03','pct_fga_03_10','pct_fga_10_16','pct_fga_16_XX','fg3a_pct_fga',
                      'fg3_pct_ast', 'fg3_pct_corner', 'pct_fg3a_corner',
                      'fg2_pct_ast', 'pct_fg2_dunk']
    OPP_ADVANCED_STATS = ['PER', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP',
                          'pct_pg', 'pct_sg', 'pct_sf', 'pct_pf', 'pct_c', 'plus_minus_on', 'plus_minus_net',
                          'tov_bad_pass', 'tov_lost_ball', 'tov_other',
                          'fouls_shooting', 'fouls_blocking', 'fouls_offensive', 'fouls_take',
                          'and1s', 'astd_pts', 'fga_blkd', 'drawn_shooting']
    POS_ALLOWED_STATS = ['FG_per100poss', 'FG%', '3P_per100poss', '3P%', 'FTr', 'TRB%', 'AST%', 'STL%', 'BLK%', 
                         'TOV%', '+/-', 'TS%', 'eFG%', 'ORtg', 'DRtg']
    # taking the differential between game stats and season average
    TEAMSHARE_STATS = ['MP', 'FGA', '3PA', 'FTA', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
    POSSESS_NORM_STATS = ["FT_per100poss", "3P_per100poss", "TOV_per100poss", "FG_per100poss", "3PA_per100poss", 
                          "DRB_per100poss", "AST_per100poss", "PF_per100poss", "PTS_per100poss", "FGA_per100poss", 
                          "STL_per100poss", "TRB_per100poss", "FTA_per100poss", "BLK_per100poss", "ORB_per100poss"]
    GAME_NORM_STATS = ['tov_bad_pass', 'tov_lost_ball', 'tov_other',
                       'fouls_shooting', 'fouls_blocking', 'fouls_offensive', 'fouls_take',
                       'and1s', 'astd_pts', 'fga_blkd', 'drawn_shooting']


    '''
    PLAYER_STATS = ["STL%", "FT", "3P", "TOV", "FG", "3PA", "DRB", "ORB%", 
                    "BLK%", "AST", "FT%", "3PAr", "PF", "PTS", "FGA", "FG%", 
                    "STL", "TRB", "TOV%", "AST%", "FTA", "eFG%", "BLK", "FTr", 
                    "+/-", "USG%", "DRB%", "TS%", "MP", "DRtg", "ORtg", "TRB%", 
                    "ORB", "3P%"]
    PER48_STATS = ["FT", "3P", "TOV", "FG", "3PA", "DRB", "AST", "PF", 
                   "PTS", "FGA", "STL", "TRB", "FTA", "BLK", "ORB"]
    TEAM_STATS = ["TOV%", "Pace", "ORtg", "FT/FGA", "ORB%", "eFG%",  "T"]
    ESPN_TEAM_STATS = ['ESPN_ORR', 'ESPN_AST', 'ESPN_PACE', 'ESPN_TS%', 
                       'ESPN_EFF FG%', 'ESPN_TO', 'ESPN_DRR', 'ESPN_REBR', 
                       'ESPN_DEF EFF', 'ESPN_OFF EFF']
    ADVANCED_STATS = ['STL%', 'fg_pct_00_03', 'pct_fga_16_XX', 'tov_bad_pass', 'PER', 
                      'pct_fga_03_10', 'WS', 'ORB%', 'OBPM', 'fg_pct_16_XX', 
                      'tov_other', '3PAr', 'fg3_pct_corner', 'plus_minus_net', 
                      'pct_fg2_dunk', 'astd_pts', 'VORP', 'TOV%', 'AST%', 'fga_blkd', 
                      'pct_fga_00_03', 'fg2_pct_ast', 'fg3a_heave', 'WS/48', 'USG%', 
                      'DRB%', 'avg_dist', 'fg2_pct', 'fg2_dunk', 'TRB%', 
                      'shots_fouled', 'DWS', 'fg_pct', 'BPM', 'fg_pct_03_10', 
                      'plus_minus_on', 'BLK%', 'DBPM', 'and1s', 
                      'fg3a_pct_fga', 'fg3_pct_ast', 'fg2a_pct_fga', 'fg3_heave', 
                      'fg_pct_10_16', 'fg3_pct', 'tov_lost_ball', 'FTr', 'pct_fga_10_16', 
                      'pct_fg3a_corner', 'tov_off_fouls', 'TS%', 'OWS', 
                      'pct_5', 'pct_4', 'pct_1', 'pct_3', 'pct_2']
    '''

    ESPN_TRACKING_STATS = [] # TODO: CANT USE THESE UNLESS WE CAN GET THEM FOR MOST PLAYERS
    INTENSITY_STATS = ['own_win_rate','opp_win_rate','own_streak','opp_streak',
                       'own_aways_in_6','opp_aways_in_6','mp_in_6']
    POSITIONS = ['c', 'pf', 'sf', 'sg', 'pg'] # keep order to keep starter position derivation working
    CAT_ONE_HOT_ON = 1
    CAT_ONE_HOT_OFF= -1

    def __init__(self, player_id, team_id, target_game=None, invalids=None):
        if isinstance(target_game, str):
            self.target_game_id = target_game
            self.upcoming_game = None
        elif isinstance(target_game, dict):
            self.upcoming_game = target_game
            self.target_game_id = None
        else:
            raise Exception("target_game not a game id or a game dict")

        self.own_team_id = team_id
        self.player_id = player_id
        self.opp_team_id = None
        self.season_start = None
        self.invalids = invalids if invalids else []


        if self.target_game_id:
            self.target_game_info = game_collection.find_one({"_id":self.target_game_id})
            if not self.target_game_info:
                raise Exception("Could not find game %s" % self.target_game_id)

            self.ts = self.target_game_info['time']

            self.invalids.extend(self.target_game_info['inactive'])

            # check that team is in the game
            self.team_game_info = team_game_collection.find_one({"game_id" : self.target_game_id,
                                                                   "team_id" : self.own_team_id})
            if not self.team_game_info:
                raise Exception("Could not find team %s in game" % self.own_team_id)
            self.opp_team_id = self.target_game_info['away_id'] if self.own_team_id == self.target_game_info['home_id'] else self.target_game_info['home_id']
            self.home_id = self.target_game_info['home_id']
            self.away_id = self.target_game_info['away_id']

            # check that player is in game
            self.player_game_info = player_game_collection.find_one({"player_id" : self.player_id, "game_id" : self.target_game_id})
            if not self.player_game_info:
                raise Exception("Could not find player %s in game" % self.player_id)

        elif self.upcoming_game:
            self.ts = self.upcoming_game['time']
            self.target_game_info = None
            self.team_game_info = None
            self.player_game_info = None
            team1_id = team_collection.find_one({"name":self.upcoming_game['home_team_name']})["_id"]
            team2_id = team_collection.find_one({"name":self.upcoming_game['away_team_name']})["_id"]
            # try to look for game
            if self.ts < datetime.now():
                found = game_collection.find_one({"time": self.ts, "teams" : [team1_id, team2_id]})
                self.target_game_info = found if found else None
            self.opp_team_id = team2_id if (self.own_team_id == team1_id) else team1_id
            self.home_id = team1_id
            self.away_id = team2_id

        # grab team rows
        self.own_team = team_collection.find_one({"all_ids": self.own_team_id})
        self.opp_team = team_collection.find_one({"all_ids": self.opp_team_id})
        self.home_team = self.own_team if (self.own_team_id == self.home_id) else self.opp_team

        # get start of season based on game timestamp (use most recent 8/1)
        self.season_start = datetime(year=self.ts.year, month=8, day=1)
        if self.season_start > self.ts:
            self.season_start = datetime(year=self.ts.year-1, month=8, day=1)

        # get all teams and locations
        results = team_collection.find({})
        self.all_teams = set()
        self.all_locations = set()
        for r in results:
            self.all_teams.add(r['_id'])
            self.all_locations |= set(r['arenas'])
        self.all_teams = list(self.all_teams)
        self.all_locations = list(self.all_locations)

    def timestamp(self):
        return self.ts

    def convertPlayerGameStatsToPer48(self, stats):
        new_stats = []
        for s in stats:
            mp = float(s['MP'])
            if mp == 0.0:
                new_stats.append(s)
            else:
                ratio = 48.0 / mp
                for k in self.PER48_STATS:
                    s[k] = s[k] * ratio
                new_stats.append(s)
        return new_stats

    def averageStats(self, stats, allowed_keys):
        trajectories = {k:[] for k in allowed_keys}
        for stat in stats:
            for k in allowed_keys:
                trajectories[k].append(stat.get(k))
        # average out the stats
        for k,v in trajectories.iteritems():
            values = [x for x in v if x is not None and x != '']
            if len(values) > 0:
                trajectories[k] = np.mean(values)
            else:
                trajectories[k] = np.nan
        return trajectories

    def varianceStats(self, stats, allowed_keys):
        trajectories = {k:[] for k in allowed_keys}
        for stat in stats:
            for k in allowed_keys:
                trajectories[k].append(stat.get(k))
        # get variance of stats
        for k,v in trajectories.iteritems():
            values = [x for x in v if x is not None and x != '']
            if len(values) > 0:
                trajectories[k] = np.var(values)
            else:
                trajectories[k] = np.nan
        return trajectories

    def serializeFeatures(self, cont_labels, cat_labels, data):
        cont_features = [data.get(k, np.nan) for k in cont_labels]
        cont_features = [v if v is not None else np.nan for v in cont_features]
        cat_features = [data.get(k, self.CAT_ONE_HOT_OFF) for k in cat_labels]
        cat_features = [v if v is not None else self.CAT_ONE_HOT_OFF for v in cat_features]
        return cont_features, cat_features

    def takeDifference(self, one, two, keys):
        if len(one) != len(two):
            raise Exception("difference arrays not same length")
        # difference
        differentials = []
        for i,reflect in enumerate(two):
            base = one[i]
            if reflect is None or base is None:
                continue
            cur_diff = {}
            for k in keys:
                base_val = base.get(k, None)
                reflect_val = reflect.get(k, None)
                if base_val is None or reflect_val is None:
                    cur_diff[k] = None
                else:
                    cur_diff[k] = base_val - reflect_val
            differentials.append(cur_diff)
        return differentials

    def chooseBiggest(self, players, player_rows, smallest=False):
        '''
        Choose heaviest player, breaking ties by height (taller)
        If reversed, choose lightest player, breaking ties by height (shorter)
        '''
        reverse = True
        if smallest:
            reverse = False

        by_weight = defaultdict(list)
        for p in players:
            by_weight[player_rows[p]['weight']].append(p)
        weight_key = sorted(by_weight.keys(), reverse=reverse)[0]
        possible_players = by_weight[weight_key]
        if len(possible_players) == 1:
            return possible_players[0]
        else:
            # break ties by height
            height_sorted = sorted(possible_players, key=lambda x: player_rows[x]['height'], reverse=reverse)
            return height_sorted[0]

    def deriveStartersFromPossible(self, possible, starter_rows):
        '''
        HEURISTIC: 
            IF NO PF, CHOOSE HEAVIEST GUY FROM SF
            IF NO SF, CHOOSE LIGHTEST GUY FROM PF
            IF NO C, CHOOSE HEAVIEST GUY FROM PF
            IF NO PG, CHOOSE LIGHTEST GUY FROM SG
            IF NO SG, CHOOSE LIGHTEST GUY FROM SF

        TIE BREAKING:
        STEP 1:
            FIND TWO POSITIONS WITH EXACT SAME POSSIBLE PLAYERS, TRY TO BREAK TIE BY PREFERRED POSITION (FIRST LISTED)
            THEN SOFT ASSIGN IT, AND SEE IF THAT ASSIGNMENT BREAKS THE TIE
        STEP 2:
        '''
        starters = {}

        # logically fill the rest
        while len(starters) < 5:
            #print starters
            starting_len = len(starters)
            empty_positions = []
            # simple filling
            for k in self.POSITIONS:
                v = possible.get(k,[])
                if k in starters:
                    continue
                remaining = list(set(v) - set(starters.values()))
                if len(remaining) == 1:
                    starters[k] = remaining[0]
                    '''
                elif len(remaining) == 0:
                    empty_positions.append(k)
            # stealing from other positions
            for k in empty_positions:
                '''
                elif len(remaining) == 0:
                    if k.lower() == 'pf':
                        eligible = list(set(possible['sf']) - set(starters.values()))
                        if eligible:
                            chosen = self.chooseBiggest(eligible, starter_rows, smallest=False)
                            starters[k] = chosen
                    elif k.lower() == 'sf':
                        pf_eligible = list(set(possible['pf']) - set(starters.values()))
                        sg_eligible = list(set(possible['sg']) - set(starters.values()))
                        if pf_eligible:
                            chosen = self.chooseBiggest(pf_eligible, starter_rows, smallest=True)
                            starters[k] = chosen
                        elif sg_eligible:
                            chosen = self.chooseBiggest(sg_eligible, starter_rows, smallest=False)
                            starters[k] = chosen
                    elif k.lower() == 'c':
                        eligible = list(set(possible['pf']) - set(starters.values()))
                        if eligible:
                            chosen = self.chooseBiggest(eligible, starter_rows, smallest=False)
                            starters[k] = chosen
                    elif k.lower() == 'pg':
                        eligible = list(set(possible['sg']) - set(starters.values()))
                        if eligible:
                            chosen = self.chooseBiggest(eligible, starter_rows, smallest=True)
                            starters[k] = chosen
                    elif k.lower() == 'sg':
                        sf_eligible = list(set(possible['sf']) - set(starters.values()))
                        pg_eligible = list(set(possible['pg']) - set(starters.values()))
                        if sf_eligible:
                            chosen = self.chooseBiggest(sf_eligible, starter_rows, smallest=True)
                            starters[k] = chosen
                        elif pg_eligible:
                            chosen = self.chooseBiggest(pg_eligible, starter_rows, smallest=False)
                            starters[k] = chosen
                # if no change yet, check for players who only have this position left to play
                if starting_len == len(starters): 
                    for r in remaining:
                        remaining_pos = list(set(starter_rows[r]['position']) - set(starters.keys()))
                        if len(remaining_pos) == 1 and remaining_pos[0] == k:
                            starters[k] = r
                            break
                # if change has been made, check if displaced players need to be pushed forward        
                if len(starters) > starting_len and k in starters:
                    displaced = list(set(remaining) - set([starters[k]]))
                    if len(displaced) == 1:
                        d = displaced[0]
                        remaining_pos = list(set(starter_rows[d]['position']) - set([k]))
                        if len(remaining_pos) == 0:
                            if k == 'pf':
                                starters['sf'] = d
                            elif k == 'sf':
                                starters['sg'] = d
                            elif k == 'sg':
                                starters['pg'] = d
            if starting_len == len(starters):
                # no change made, we are in infinite loop, start breaking ties
                # find positions that share same value
                pos_unfilled = list(set(self.POSITIONS) - set(starters.keys()))
                pos_remaining = {_ : (set(possible[_]) - set(starters.values())) for _ in pos_unfilled}
                players_remaining = list(set(starter_rows.keys()) - set(starters.values()))
                broken = False
                pairs = []
                for a,b in itertools.combinations(pos_remaining.keys(), 2):
                    if pos_remaining[a] == pos_remaining[b]:
                        pairs.append((a,b))
                # fill players who are only listed in that one position, if tie is still not broken
                if not broken:
                    for pid in players_remaining:
                        listed = starter_rows[pid]['position']
                        listed_avai = list(set(listed) - set(starters.keys()))
                        if len(listed) == 1 and listed[0] not in starters:
                            starters[listed[0]] = pid
                            broken = True
                            break
                        elif len(listed_avai) == 1 and listed_avai[0] not in starters:
                            starters[listed_avai[0]] = pid
                            broken = True
                            break
                # try to break pairs by first listed position
                if not broken:
                    for pair in pairs:
                        # first try to break up pair by preferred position
                        pos1, pos2 = pair
                        players = pos_remaining[pos1]
                        if len(players) == 2:
                            pref_assignment = defaultdict(list)
                            for p in players:
                                pref = starter_rows[p]['position'][0]
                                pref_assignment[pref].append(p)
                            if set(pref_assignment.keys()) == set([pos1, pos2]):
                                for k,v in pref_assignment.items():
                                    starters[k] = v[0]
                                broken = True
                                break
                # if not broken, try to break sf/pf and sf/sg pairs by weight, then height (heavier to pf or sf)
                if not broken:
                    for pos1, pos2 in pairs:
                        players = pos_remaining[pos1]
                        if set([pos1, pos2]) == set(['sf','pf']):
                            chosen = self.chooseBiggest(players, starter_rows, smallest=False)
                            starters['pf'] = chosen
                        elif set([pos1, pos2]) == set(['sg', 'pg']):
                            chosen = self.chooseBiggest(players, starter_rows, smallest=True)
                            starters['pg'] = chosen
                        elif set([pos1, pos2]) == set(['sg', 'sf']):
                            chosen = self.chooseBiggest(players, starter_rows, smallest=False)
                            starters['sf'] = chosen
                        elif set([pos1, pos2]) == set(['pf', 'c']):
                            chosen = self.chooseBiggest(players, starter_rows, smallest=False)
                            starters['c'] = chosen
        return starters

    def getStarters(self):
        '''
        Position choosing is kind of iffy

        Possible infinite loop with assigning starters to positions
        '''
        if self.target_game_info is not None:
            # find starters
            home_starters = self.target_game_info['home_starters']
            away_starters = self.target_game_info['away_starters']
            home_starter_rows = {k: player_collection.find_one({"_id" : k}) for k in home_starters}
            away_starter_rows = {k: player_collection.find_one({"_id" : k}) for k in away_starters}
            # find home starter positions
            home_by_pos = defaultdict(list)
            away_by_pos = defaultdict(list)
            preferred_pos = {}
            for s in home_starters:
                p_row = player_collection.find_one({"_id" : s})
                if not p_row:
                    raise Exception("%s starter not found" % s)
                preferred_pos[s] = p_row['position'][0]
                for p_pos in p_row['position']:
                    home_by_pos[p_pos].append(s)
            for s in away_starters:
                p_row = player_collection.find_one({"_id" : s})
                if not p_row:
                    raise Exception("%s starter not found" % s)
                preferred_pos[s] = p_row['position'][0]
                for p_pos in p_row['position']:
                    away_by_pos[p_pos].append(s)
            possible_home_starters = dict(home_by_pos)
            possible_away_starters = dict(away_by_pos)

            print "Possible Home Starters: %s" % possible_home_starters
            print "Possible Away Starters: %s" % possible_away_starters

            home_starters = self.deriveStartersFromPossible(possible_home_starters, home_starter_rows)
            away_starters = self.deriveStartersFromPossible(possible_away_starters, away_starter_rows)
            own_starters = home_starters if self.own_team_id == self.home_id else away_starters
            opp_starters = home_starters if self.opp_team_id == self.home_id else away_starters

            # find bench players
            opp_played = player_game_collection.find({"game_id" : self.target_game_info['_id'], "player_team" : self.opp_team_id})
            own_played = player_game_collection.find({"game_id" : self.target_game_info['_id'], "player_team" : self.own_team_id})
            opp_pid = [_['player_id'] for _ in opp_played if _['MP'] > 0.0]
            own_pid = [_['player_id'] for _ in own_played if _['MP'] > 0.0]

            opp_bench = list(set(opp_pid) - set(opp_starters.values()) - set(self.invalids))
            own_bench = list(set(own_pid) - set(own_starters.values()) - set(self.invalids))

            # sort the bench
            mp_last_game = {}
            opp_bench_by_pos = defaultdict(list)
            own_bench_by_pos = defaultdict(list)
            for pid in opp_bench:
                last_game = player_game_collection.find({"player_id" : pid, "player_team" : self.opp_team_id, "game_time" : {"$lt": self.ts}}, sort=[("game_time",-1)])[0]
                player_row = player_collection.find_one({"_id" : pid})
                mp_last_game[pid] = last_game.get('MP',0.0) if last_game is not None else 0.0
                for p in player_row['position']:
                    opp_bench_by_pos[p].append(pid)
            for pid in own_bench:
                last_game = player_game_collection.find({"player_id" : pid, "player_team" : self.own_team_id, "game_time" : {"$lt": self.ts}}, sort=[("game_time",-1)])[0]
                player_row = player_collection.find_one({"_id" : pid})
                mp_last_game[pid] = last_game.get('MP',0.0) if last_game is not None else 0.0
                for p in player_row['position']:
                    own_bench_by_pos[p].append(pid)
            opp_bench = dict(opp_bench_by_pos)
            own_bench = dict(own_bench_by_pos)
            for k,v in opp_bench.iteritems():
                opp_bench[k] = list(sorted(v, key=lambda x: mp_last_game[x], reverse=True))
            for k,v in own_bench.iteritems():
                own_bench[k] = list(sorted(v, key=lambda x: mp_last_game[x], reverse=True))
        else:
            # get espn depth chart
            charts = depth_collection.find({"time" : {"$gt" : self.season_start}})
            chosen_chart = sorted(charts, key=lambda x: abs(self.ts-x['time']))[0]
            chart_invalids = chosen_chart['invalids']
            chart_depth = chosen_chart['stats']
            own_bench = chart_depth[self.own_team_id]
            opp_bench = chart_depth[self.opp_team_id]
            own_starters = {}
            for k,v in own_bench:
                own_starters[k] = v.pop(0)
            for k,v in opp_bench:
                opp_starters[k] = v.pop(0)

            self.invalids.extend(chart_invalids)
            
        # switch out invalid starters for bench player if needed
        for k,v in opp_starters.iteritems():
            if v in self.invalids:
                while True:
                    chosen = opp_bench[k].pop(0)
                    if chosen not in opp_starters.values():
                        break
                opp_starters[k] = chosen
        for k,v in own_starters.iteritems():
            if v in self.invalids:
                while True:
                    chosen = own_bench[k].pop(0)
                    if chosen not in own_starters.values():
                        break
                own_starters[k] = chosen


        print "Own Starters: %s" % own_starters
        print "Own Bench: %s" % own_bench
        print "Opp Starters: %s" % opp_starters
        print "Opp Bench: %s" % opp_bench

        self.own_roster = list(set(own_starters.values()) | set([x for _ in own_bench.values() for x in _]))
        self.opp_roster = list(set(opp_starters.values()) | set([x for _ in opp_bench.values() for x in _]))
        self.own_starters = own_starters
        self.own_bench = own_bench
        self.opp_starters = opp_starters
        self.opp_bench = opp_bench

        if self.player_id not in self.own_roster:
            raise Exception("Player not playing in game...")

        # find player row
        self.player_row = player_collection.find_one({"_id" : self.player_id})

        # find player position
        self.pos = None
        self.starting = self.CAT_ONE_HOT_OFF
        for k,v in self.own_starters.iteritems():
            if self.player_id == v:
                self.pos = k
                self.starting = self.CAT_ONE_HOT_ON
                break
        if self.pos == None:
            self.pos = self.player_row['position'][0]

    def getAdvancedStatsForPlayer(self, player_id, team_id, filter_keys):
        advanced_rows = list(advanced_collection.find({"player_id" : player_id, "team_id" : team_id, "time" : {"$gt" : self.season_start, "$lt" : self.season_start + timedelta(365)}}))
        if len(advanced_rows) == 0:
            raise Exception("Could not find valid advanced stats row for %s, %s, %s" % (player_id, team_id, self.season_start))
        chosen_advanced_row = sorted(advanced_rows, key=lambda x: abs(self.ts - x['time']))[0]
        # normalize by 48 minutes for the specified keys
        mp = float(chosen_advanced_row["MP"])
        by36 = mp/36.0
        for k in self.GAME_NORM_STATS:
            if k in chosen_advanced_row:
                normed = chosen_advanced_row.get(k,0.0) / by36
                chosen_advanced_row[k] = normed
        # filter keys
        player_advanced_stats = {k: chosen_advanced_row[k] for k in filter_keys}
        return player_advanced_stats

    def getPlayerStats(self):
        '''
        For the player:
         -> season long advanced/shooting/play by play stats
         -> running avg stats
         -> running avg variance
         -> player position, height, weight, handedness
         -> TEAM EFFECTS: percentage of team sums

        For the opposing team:
         -> player specific stats for the starter and backup for the player's position
            -> running avg stats
            -> season long advanced/play by play stats
            -> height and size difference
         -> earned team stats by position
            -> defensive/offensive stats earned (per position) for a running window, per 48
         -> given up stats by position
            -> defensive/offensive stats given up (per position) for a running window, per 48

        '''
        player_stats = list(player_game_collection.find({"player_id": self.player_id, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, limit=10, sort=[("game_time",-1)]))
        own_last_10_games = list(game_collection.find({"teams": self.own_team_id, "time" : {"$lt" : self.ts, "$gt" : self.season_start}}, limit=10, sort=[("time",-1)]))
        opp_last_10_games = list(game_collection.find({"teams": self.opp_team_id, "time" : {"$lt" : self.ts, "$gt" : self.season_start}}, limit=10, sort=[("time",-1)]))

        # check that we have enough statistics
        if len(own_last_10_games) < 2:
            raise Exception("Not enough previous games for own %s" % self.own_team_id)
        if len(opp_last_10_games) < 2:
            raise Exception("Not enough previous games for opp %s" % self.opp_team_id)
        if len(player_stats) < 2:
            raise Exception("Not enough previous games for player %s" % self.player_id)

        # get player stats running avg stats
        self.player_running_avg = self.averageStats(player_stats, self.PLAYER_STATS)
        # get stat variances
        self.player_running_variance = self.varianceStats(player_stats, self.PLAYER_STATS)

        # one hot encode the position
        '''
        self.pos_encoding = {}
        for _ in self.POSITIONS:
            if _ in self.player_row['position']:
                self.pos_encoding[_] = self.CAT_ONE_HOT_ON
            else:
                self.pos_encoding[_] = self.CAT_ONE_HOT_OFF
        '''

        height = self.player_row['height']
        weight = self.player_row['weight']
        shoots = self.CAT_ONE_HOT_ON if self.player_row.get('shoots','R') == 'R' else self.CAT_ONE_HOT_OFF

        self.physical_stats = {'height' : height,
                               'weight' : weight,
                               'shoots' : shoots}

        '''
        # get espn stats
        espn_rows = espn_player_stat_collection.find({"player_id" : self.player_id, "time" : {"$gt" : self.season_start}})
        chosen_espn_row = sorted(espn_rows, key=lambda x: abs(self.ts - x['time']))[0]
        self.player_espn_stats = {k: chosen_espn_row[k] for k in self.ESPN_TRACKING_STATS}
        '''

        # get advanced stats
        self.player_advanced_stats = self.getAdvancedStatsForPlayer(self.player_id, self.own_team_id, self.ADVANCED_STATS)

        # get team effects (stat fractions)
        # MP, FGA, 3PA, FTA, TRB, AST, STL, BLK, TOV
        teamshare_stats = defaultdict(list)
        for g in own_last_10_games:
            gid = g['_id']
            g_players = list(player_game_collection.find({"game_id": gid, "player_team": self.own_team_id}))
            p_row = player_game_collection.find_one({"game_id": gid, "player_id": self.player_id})
            if p_row is None:
                p_row = {}
            for s in self.TEAMSHARE_STATS:
                total = sum([float(x.get(s,0.0)) for x in g_players])
                if total > 0.0:
                    fraction = float(p_row.get(s,0.0)) / total
                    teamshare_stats[s].append(fraction)
        self.teamshare = {k: np.mean(v) for k,v in teamshare_stats.items()}

        # find top 2 opp team player matchups

        # HEURISTIC: FINDING THE TWO OPP PLAYERS MOST LIKELY TO PLAY/SWITCH ON A POSITION ON THE COURT
        self.pos_starter = self.opp_starters[self.pos]
        self.pos_backup = None
        if len(self.opp_bench.get(self.pos,[])) > 0:
            self.pos_backup = self.opp_bench[self.pos][0]
        if self.pos_backup is None:
            if self.pos == 'c' and len(self.opp_bench.get('pf', [])) > 0:
                self.pos_backup = self.opp_bench['pf'][0]
            elif self.pos == 'pf' and len(self.opp_bench.get('sf', [])) > 0:
                self.pos_backup = self.opp_bench['sf'][0]
            elif self.pos == 'sf' and len(self.opp_bench.get('sg', [])) > 0:
                self.pos_backup = self.opp_bench['sg'][0]
            elif self.pos == 'pg' and len(self.opp_bench.get('sg', [])) > 0:
                self.pos_backup = self.opp_bench['sg'][0]
            elif self.pos == 'sg' and len(self.opp_bench.get('pg', [])) > 0:
                self.pos_backup = self.opp_bench['pg'][0]
        if self.pos_backup is None:
            raise Exception("Could not find a valid backup for %s" % self.pos)

        starter_stats = list(player_game_collection.find({"player_id" : self.pos_starter, "game_time" : {"$lt" : self.ts, "$gt" : self.season_start}},
                                                         sort=[("game_time",-1)], limit=10))
        backup_stats = list(player_game_collection.find({"player_id" : self.pos_backup, "game_time" : {"$lt" : self.ts, "$gt" : self.season_start}},
                                                        sort=[("game_time",-1)], limit=10))
        if len(starter_stats) < 1:
            raise Exception("No last games found for %s" % self.pos_starter)
        if len(backup_stats) < 1:
            raise Exception("No last games found for %s" % self.pos_backup)

        starter_row = player_collection.find_one({"_id" : self.pos_starter})
        backup_row = player_collection.find_one({"_id" : self.pos_backup})
        starter_height_diff = height - starter_row['height']
        starter_weight_diff = weight - starter_row['weight']
        backup_height_diff = height - backup_row['height']
        backup_weight_diff = weight - backup_row['weight']

        # get advanced stats for opposing starter and backup
        self.pos_starter_advanced_stats = self.getAdvancedStatsForPlayer(self.pos_starter, self.opp_team_id, self.OPP_ADVANCED_STATS)
        self.pos_backup_advanced_stats = self.getAdvancedStatsForPlayer(self.pos_backup, self.opp_team_id, self.OPP_ADVANCED_STATS)

        self.opp_starter_avgs = self.averageStats(starter_stats, self.OPP_PLAYER_STATS)
        self.opp_backup_avgs = self.averageStats(backup_stats, self.OPP_PLAYER_STATS)
        self.opp_starter_avgs['height_difference'] = starter_height_diff
        self.opp_starter_avgs['weight_difference'] = starter_weight_diff
        self.opp_backup_avgs['height_difference'] = backup_height_diff
        self.opp_backup_avgs['weight_difference'] = backup_weight_diff

        self.calculatePositionStats(opp_last_10_games)


    def calculatePositionStats(self, opp_last_10_games):
        '''
        find earned/given up stats per position, by differential between game stats and season average
        '''
        season_stats = {}
        player_blacklist = set()
        self.position_stats = {}
        for position in self.POSITIONS:
            earned_stats = []
            givenup_stats = []
            for game_row in opp_last_10_games:
                game_id = game_row['_id']
                opp_opp_id = game_row['home_id'] if game_row['home_id'] != self.opp_team_id else game_row['away_id']
                earned_players = player_game_collection.find({"game_id" : game_id, "player_team": self.opp_team_id})
                givenup_players = player_game_collection.find({"game_id" : game_id, "player_team": opp_opp_id})
                for ep in earned_players:
                    # get player row
                    pid = ep['player_id']
                    p_row = player_collection.find_one({"_id" : pid})
                    if p_row is None:
                        continue
                    if position in p_row['position'] and ep['MP'] > 10.0:
                        earned_stats.append(ep)
                        # get season stats if necessary
                        if pid not in season_stats and pid not in player_blacklist:
                            player_team = ep['player_team']
                            pid_season = list(player_game_collection.find({"player_team" : player_team, "player_id" : pid, "game_time" : {"$gt" : self.season_start}}))
                            if len(pid_season) < 3:
                                player_blacklist.add(pid)
                            else:
                                season_stats[pid] = self.averageStats(pid_season, self.POS_ALLOWED_STATS)
                for gp in givenup_players:
                    pid = gp['player_id']
                    p_row = player_collection.find_one({"_id" : pid})
                    if p_row is None:
                        continue
                    p_position = p_row['position'][0]
                    if position in p_row['position'] and gp['MP'] > 10.0:
                        givenup_stats.append(gp)
                        # get season stats if necessary
                        if pid not in season_stats and pid not in player_blacklist:
                            player_team = gp['player_team']
                            pid_season = list(player_game_collection.find({"player_team" : player_team, "player_id" : pid, "game_time" : {"$gt" : self.season_start}}))
                            if len(pid_season) < 3:
                                player_blacklist.add(pid)
                                print "Not enough games in season %s for %s on %s" % (self.season_start, pid, player_team)
                            else:
                                season_stats[pid] = self.averageStats(pid_season, self.POS_ALLOWED_STATS)
            givenup_ratios = []
            for givenup in givenup_stats:
                pid = givenup['player_id']
                if pid in player_blacklist:
                    continue
                pid_season_avg = season_stats[pid]
                ratios = {}
                for k in self.POS_ALLOWED_STATS:
                    num = givenup[k]
                    denum = pid_season_avg[k]
                    try:
                        num = float(num)
                        denum = float(denum)
                    except ValueError as e:
                        ratios[k] = 1.0
                        continue
                    ratios[k] = num/denum if denum > 0.0 else 1.0
                givenup_ratios.append(ratios)
            ratio_averages = self.averageStats(givenup_ratios, self.POS_ALLOWED_STATS)
            for k,v in ratio_averages.iteritems():
                self.position_stats['%s_givenup_%s' % (position, k)] = v

    def getTeamStats(self):
        '''
        query for all the necessary team stats, which includes:
            - espn team stats (closest to game time)
            - running 5 game team stat averages
            - season averages
            - most recent game played
            - previous matchups in the season
        '''
        # find previous matchups between the two teams
        self.prev_matchup_games = list(game_collection.find({"teams": [self.own_team_id, self.opp_team_id], "time": {"$gt" : self.season_start}}))
        self.own_prev_matchup_stats = [team_game_collection.find_one({"team_id": self.own_team_id, "game_id": g['_id']}) for g in self.prev_matchup_games]
        self.opp_prev_matchup_stats = [team_game_collection.find_one({"team_id": self.opp_team_id, "game_id": g['_id']}) for g in self.prev_matchup_games]

        # find espn team stats
        espnstat = list(espn_stat_collection.find({"time": {"$gt" : self.season_start}}))
        sorted_espnstat = sorted(espnstat, key=lambda x: abs(self.ts-x['time']))
        recent_espnstat = sorted_espnstat[0]['stats']
        self.own_espn_team_stats = {k:v for k,v in recent_espnstat[self.own_team_id].iteritems() if k in self.TEAM_ESPN_STATS}
        self.opp_espn_team_stats = {k:v for k,v in recent_espnstat[self.opp_team_id].iteritems() if k in self.TEAM_ESPN_STATS}

        # own season/running avg team stats ( + reflection)
        own_results = list(team_game_collection.find({"team_id": self.own_team_id, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, 
                                            sort=[("game_time",-1)]))
        if len(own_results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % self.own_team_id)
        own_reflection_results = [team_game_collection.find_one({"game_id": g['game_id'], "team_id": {"$ne": self.own_team_id}}) for g in own_results]
        
        own_results = self.calculateAdditionalTeamStats(own_results, own_reflection_results)

        # difference
        #own_differentials = self.takeDifference(own_results, own_reflection_results, self.TEAM_STATS)

        self.own_most_recent_game = game_collection.find_one({"_id" : own_results[0]['game_id']})
        self.own_last_10 = copy.deepcopy(own_results[:10])
        self.own_gametimes = [o['game_time'] for o in own_results[:5]]
        self.own_team_season_avgs = self.averageStats(own_results, self.TEAM_STATS)
        self.own_team_running_avgs = self.averageStats(own_results[:10], self.TEAM_STATS)
        #self.own_reflection_season_avgs = self.averageStats(own_reflection_results, self.TEAM_STATS)
        #self.own_reflection_running_avgs = self.averageStats(own_reflection_results[:10], self.TEAM_STATS)
        #self.own_differentials_avgs = self.averageStats(own_differentials[:10], self.TEAM_STATS)

        # opp season/running avg team stats ( + reflection)
        opp_results = list(team_game_collection.find({"team_id": self.opp_team_id, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, 
                                            sort=[("game_time",-1)]))
        if len(opp_results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % self.opp_team_id)

        opp_reflection_results = [team_game_collection.find_one({"game_id": g['game_id'], "team_id": {"$ne": self.opp_team_id}}) for g in opp_results]
       
        opp_results = self.calculateAdditionalTeamStats(opp_results, opp_reflection_results)

        # difference
        #opp_differentials = self.takeDifference(opp_results, opp_reflection_results, self.TEAM_STATS)

        self.opp_most_recent_game = game_collection.find_one({"_id" : opp_results[0]['game_id']})
        self.opp_last_10 = copy.deepcopy(opp_results[:10])
        self.opp_gametimes = [o['game_time'] for o in opp_results[:5]]
        self.opp_team_season_avgs = self.averageStats(opp_results, self.TEAM_STATS)
        self.opp_team_running_avgs = self.averageStats(opp_results[:10], self.TEAM_STATS)
        #self.opp_reflection_season_avgs = self.averageStats(opp_reflection_results, self.TEAM_STATS)
        #self.opp_reflection_running_avgs = self.averageStats(opp_reflection_results[:10], self.TEAM_STATS)
        #self.opp_differentials_avgs = self.averageStats(opp_differentials[:10], self.TEAM_STATS)

    def calculateAdditionalTeamStats(self, own_results, opp_results):
        '''
        Calculating
            DRB% = 100 - (opponent's ORB%)
            DRtg = opponent's ORtg
            NetRtg = ORtg - DRtg
            T_net = (own T) - (opp T)
        Appending
            opp_eFG%
            opp_TOV%
            opp_FT/FGA
        '''
        if len(own_results) != len(opp_results):
            raise Exception("own team results and opp team results not same length")
        results = []
        for own,opp in zip(own_results, opp_results):
            new_row = copy.deepcopy(own)
            new_row['opp_eFG%'] = float(opp['eFG%'])
            new_row['opp_TOV%'] = float(opp['TOV%'])
            new_row['opp_FT/FGA'] = float(opp['FT/FGA'])
            new_row['DRB%'] = 100.0 - float(opp['ORB%'])
            new_row['DRtg'] = float(opp['ORtg'])
            new_row['NetRtg'] = float(own['ORtg']) - float(opp['ORtg'])
            new_row['T_net'] = float(own['T']) - float(opp['T'])
            results.append(new_row)
        return results

    def runEncoderFeatures(self):
        """

        TEAM FEATURES (OWN,OPP):
            - espn team features (shortest temporal distance) NEED PREVIOUS SEASON ESPN TEAM STATS
            - running avg team features (avg over 5 games, include running avgs of opponent teams faced)
            - record features (win percentage, aways in last 5, wins in last 10)

        PLAYER FEATURES:
            - see player features function

        LOCATION FEATURES:
            - location
            - own team and opp team
            - b2b for own team and opp team (CHANGE TO DAYS REST)

        ?? PREVIOUS MATCHUP FEATURES (OWN, OPP):
            - same as TEAM FEATURES (without espn team features) but only averaging over previous team matchups

        TREND FEATURES:
            - for each team:
                - (season - running avg) for team stats
        """
        self.getStarters()
        self.getTeamStats()
        self.getPlayerStats()

        # UNUSED: self.matchupFeatures
        fns = [self.playerFeatures, self.teamFeatures, self.locationFeatures, self.recordFeatures, self.trendFeatures]
        cat_labels_all = []
        cat_features_all = []
        cont_labels_all = []
        cont_features_all = []
        cat_feat_splits_all = []
        for fn in fns:
            cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = fn()
            cat_labels_all += cat_labels
            cat_features_all += cat_features
            cont_labels_all += cont_labels
            cont_features_all += cont_features
            cat_feat_splits_all += cat_feat_splits

        return cat_labels_all, cat_features_all, cont_labels_all, cont_features_all, cat_feat_splits_all

    def playerFeatures(self):
        cont_data = {'physical' : copy.deepcopy(self.physical_stats),
                     'advanced' : copy.deepcopy(self.player_advanced_stats),
                     'teamshare' : copy.deepcopy(self.teamshare),
                     'opp_starter' : copy.deepcopy(self.opp_starter_avgs),
                     'opp_backup' : copy.deepcopy(self.opp_backup_avgs),
                     'opp_starter_advanced': copy.deepcopy(self.pos_starter_advanced_stats),
                     'opp_backup_advanced': copy.deepcopy(self.pos_backup_advanced_stats),
                     'own' : copy.deepcopy(self.player_running_avg),
                     'own_var' : copy.deepcopy(self.player_running_variance),
                     'opp' : copy.deepcopy(self.position_stats)
                     }
        cat_data = {}

        # flatten
        top_keys = cont_data.keys()
        for k in top_keys:
            for k_2,v in cont_data[k].iteritems():
                cont_data["%s_%s" % (k,k_2)] = v
            cont_data.pop(k,None)

        top_keys = cat_data.keys()
        for k in top_keys:
            for k_2,v in cat_data[k].iteritems():
                cat_data["%s_%s" % (k,k_2)] = v
            cat_data.pop(k,None)

        # serialize
        cont_labels = list(sorted(cont_data.keys()))
        cat_labels = list(sorted(cat_data.keys()))
        cat_feature_splits = []

        data = cont_data
        data.update(cat_data)
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)


    def teamFeatures(self):
        '''
        for each team: espn stats, running avg, running opp average
        '''
        data = {'own_espn_team_stats': copy.deepcopy(self.own_espn_team_stats),
                'opp_espn_team_stats': copy.deepcopy(self.opp_espn_team_stats),
                'own_team_running_avgs': copy.deepcopy(self.own_team_running_avgs),
                'opp_team_running_avgs': copy.deepcopy(self.opp_team_running_avgs)}

        # flatten data
        top_keys = data.keys()
        for k in top_keys:
            for k_2,v in data[k].iteritems():
                data["%s_%s" % (k,k_2)] = v
            data.pop(k,None)

        # serialize
        cont_labels = list(sorted(data.keys()))
        cat_labels = []
        cat_feature_splits = []
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)

    def trendFeatures(self):
        own_trend_diff = self.takeDifference([self.own_team_season_avgs], [self.own_team_running_avgs], self.TEAM_STATS)[0]
        opp_trend_diff = self.takeDifference([self.opp_team_season_avgs], [self.opp_team_running_avgs], self.TEAM_STATS)[0]

        data = {'own_recent_trend': own_trend_diff,
                'opp_recent_trend': opp_trend_diff}

        # flatten data
        top_keys = data.keys()
        for k in top_keys:
            for k_2,v in data[k].iteritems():
                data["%s_%s" % (k,k_2)] = v
            data.pop(k,None)

        # serialize
        cont_labels = list(sorted(data.keys()))
        cat_labels = []
        cat_feature_splits = []
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)

    def locationFeatures(self):
        game_location = self.target_game_info['location'].split(',')[0].strip()
        teams_one_hot = {x: self.CAT_ONE_HOT_OFF for x in self.all_teams}
        location_one_hot = {x: self.CAT_ONE_HOT_OFF for x in self.all_locations}

        teams_one_hot[self.opp_team_id] = 1
        teams_one_hot[self.own_team_id] = 1
        location_one_hot[game_location] = 1

        data = {}
        for k,v in teams_one_hot.iteritems():
            data['team_%s' % k] = v
        for k,v in location_one_hot.iteritems():
            data['location_%s' % k] = v

        game_day = datetime(year=self.ts.year,month=self.ts.month,day=self.ts.day)
        own_last_game_days = [datetime(year=x.year,month=x.month,day=x.day) for x in self.own_gametimes]
        opp_last_game_days = [datetime(year=x.year,month=x.month,day=x.day) for x in self.opp_gametimes]
        own_game_deltas = [(game_day - x).days for x in own_last_game_days]
        opp_game_deltas = [(game_day - x).days for x in opp_last_game_days]

        own_days_since_last = own_game_deltas[0]
        opp_days_since_last = opp_game_deltas[0]
        data['own_days_since_last'] = own_days_since_last
        data['opp_days_since_last'] = opp_days_since_last

        '''
        own_game_index = [0] * 4
        opp_game_index = [0] * 4
        for d in own_game_deltas:
            if d < 5:
                own_game_index[d-1] = 1
        for d in opp_game_deltas:
            if d < 5:
                opp_game_index[d-1] = 1

        data['own_b2b'] = self.CAT_ONE_HOT_ON if (sum(own_game_index[:1]) >= 1) else self.CAT_ONE_HOT_OFF
        data['own_2of3'] = self.CAT_ONE_HOT_ON if (sum(own_game_index[:2]) >= 1) else self.CAT_ONE_HOT_OFF
        data['own_3of4'] = self.CAT_ONE_HOT_ON if (sum(own_game_index[:3]) >= 2) else self.CAT_ONE_HOT_OFF
        data['own_4of5'] = self.CAT_ONE_HOT_ON if (sum(own_game_index[:4]) >= 3) else self.CAT_ONE_HOT_OFF

        data['opp_b2b'] = self.CAT_ONE_HOT_ON if (sum(opp_game_index[:1]) >= 1) else self.CAT_ONE_HOT_OFF
        data['opp_2of3'] = self.CAT_ONE_HOT_ON if (sum(opp_game_index[:2]) >= 1) else self.CAT_ONE_HOT_OFF
        data['opp_3of4'] = self.CAT_ONE_HOT_ON if (sum(opp_game_index[:3]) >= 2) else self.CAT_ONE_HOT_OFF
        data['opp_4of5'] = self.CAT_ONE_HOT_ON if (sum(opp_game_index[:4]) >= 3) else self.CAT_ONE_HOT_OFF
        '''
        # serialize
        cont_labels = ['own_days_since_last', 'opp_days_since_last']
        cat_labels = ['team_%s' % k for k in self.all_teams] + ['location_%s' % k for k in self.all_locations]
        cat_feature_splits = [len(self.all_teams), len(self.all_locations)]
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)

    def recordFeatures(self):
        '''
        win percentage, aways in last 5, last 10 record, streaks
        away or home for the own team
        '''
        # calculate number of aways in last 5
        own_aways = len([_ for _ in self.own_last_10[:5] if _.get('location') == 'Away'])
        opp_aways = len([_ for _ in self.opp_last_10[:5] if _.get('location') == 'Away'])

        # calculate last 10 wins
        own_last_10_wins = 0
        for g in self.own_last_10:
            pts = g['T']
            other_team_game = team_game_collection.find_one({"game_id" : g['game_id'], "team_id" : {"$ne" : g['team_id']}})
            other_pts = other_team_game['T']
            if pts > other_pts:
                own_last_10_wins += 1
        opp_last_10_wins = 0
        for g in self.opp_last_10:
            pts = g['T']
            other_team_game = team_game_collection.find_one({"game_id" : g['game_id'], "team_id" : {"$ne" : g['team_id']}})
            other_pts = other_team_game['T']
            if pts > other_pts:
                opp_last_10_wins += 1

        # parse records and streaks
        own_recent_loc = 'home' if (self.own_team['_id'] == self.own_most_recent_game['home_id']) else 'away'
        own_record = self.own_most_recent_game['%s_record' % own_recent_loc]
        own_streak = self.own_most_recent_game['%s_streak' % own_recent_loc]

        opp_recent_loc = 'home' if (self.opp_team['_id'] == self.opp_most_recent_game['home_id']) else 'away'
        opp_record = self.opp_most_recent_game['%s_record' % opp_recent_loc]
        opp_streak = self.opp_most_recent_game['%s_streak' % opp_recent_loc]

        current_location = 1 if (self.home_team['_id'] == self.own_team['_id']) else -1

        if own_record is None:
            own_winning_percent = 1.0
            own_streak = 0.0
        else:
            own_record_parts = own_record.split('-')
            own_wins = int(own_record_parts[0].strip())
            own_losses = int(own_record_parts[1].strip())
            own_winning_percent = own_wins / float(own_losses + own_wins)
            if 'Lost' in own_streak:
                own_streak = -int(own_streak.replace('Lost ','').strip())
            else:
                own_streak = int(own_streak.replace('Won ','').strip())
        if opp_record is None:
            opp_winning_percent = 1.0
            opp_streak = 0.0
        else:
            opp_record_parts = opp_record.split('-')
            opp_wins = int(opp_record_parts[0].strip())
            opp_losses = int(opp_record_parts[1].strip())
            opp_winning_percent = opp_wins / float(opp_losses + opp_wins)
            if 'Lost' in opp_streak:
                opp_streak = -int(opp_streak.replace('Lost ','').strip())
            else:
                opp_streak = int(opp_streak.replace('Won ','').strip())

        # load feature dictionary
        data = {'own_win_rate': own_winning_percent,
                'opp_win_rate': opp_winning_percent,
                'own_streak': own_streak,
                'opp_streak': opp_streak,
                'own_aways': own_aways,
                'opp_aways': opp_aways,
                'own_last_10_wins': own_last_10_wins,
                'opp_last_10_wins': opp_last_10_wins,
                'own_home_or_away': current_location, # TEAM-SUBJECTIVE STAT
                'starting': self.starting
                }

        # serialize
        cont_labels = list(sorted(data.keys()))
        cat_labels = []
        cat_feature_splits = []
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)

if __name__=="__main__":
    
    arg = {'player_id': 'warretj01', 'team_id': 'PHO', 'target_game': '201502250DEN'}
    invalids = []
    
    fe = NBAFeatureExtractor(arg['player_id'], arg['team_id'], target_game=arg['target_game'], invalids=invalids)
    cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = fe.runEncoderFeatures()
    
    print cat_feat_splits

    for l,v in zip(cat_labels, cat_features):
        print "%s: %s" % (l,v)

    print '\n'
    for l,v in zip(cont_labels, cont_features):
        print "%s: %s" % (l,v)
    
    sys.exit(1)
    
    args = findAllTrainingGames(pid=None, limit=None, time=None, min_count=None)
    for arg in args:
        try:
            print arg
            fe = NBAFeatureExtractor(arg['player_id'], arg['team_id'], target_game=arg['target_game'])
            cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = fe.runEncoderFeatures()
        except Exception as e:
            traceback.print_exc()


