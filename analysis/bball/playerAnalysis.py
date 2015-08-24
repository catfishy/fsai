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
import csv
import os
import multiprocessing as mp
import re

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

from statsETL.db.mongolib import *


def findAllTrainingGames(pid=None, limit=None, time=None, min_count=None, mp_limit=8):
    # find all training games
    query = {}
    if pid:
        query["player_id"] = pid
    if time:
        query["game_time"] = {"$gt" : time}
    print "Training game query: %s" % query
    playergames = player_game_collection.find(query)

    args = []
    for game in playergames:
        if game['MP'] < float(mp_limit):
            continue
        # create args
        new_arg = {}
        new_arg['player_id'] = str(game['player_id'])
        new_arg['team_id'] = str(game['player_team'])
        new_arg['target_game'] = str(game['game_id'])
        args.append(new_arg)
        # limit
        if limit and len(args) == limit:
            break

    # check above min
    if min_count and len(args) < min_count:
        raise Exception("Couldn't find more than %s games for %s" % (min_count, pid))
    
    return args


class NBAFeatureExtractor(object):

    '''
    TODO: REWRITE OPP PLAYER STARTER/BACKUP SELECTION TO TAKE PLAYER POSITION PERCENTAGES INTO ACCOUNT
    TODO: NORMALIZE GAME NORM ADVANCED STATS BY NUMBER OF GAMES IN CRAWLER
    '''


    # TEAM STATS
    TEAM_ESPN_STATS = ['AST', 'TS%', 'REBR']
    TEAM_STATS = ['Pace', 'ORtg', 'DRtg','NetRtg', 'eFG%', 'ORB%', 'DRB%','TOV%', 
                  'FT/FGA', 'T', 'T_net', 'opp_eFG%', 'opp_TOV%', 'opp_FT/FGA', "Ast'd", "%Ast'd",
                  "2PA", "3PA", "2P%", "3P%", "opp_2PA", "opp_3PA", "opp_2P%", "opp_3P%"]
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
                    "STL_per100poss", "TRB_per100poss", "FTA_per100poss", "BLK_per100poss", "ORB_per100poss", 
                    'StopPercent', 'Stops', 'PProd', 'FloorPercent', 'TotPoss']
    # PTS/FGA = PTS / FGA
    # FTA/FGA = FTA/FGA ::: REDUNDANT (FTr)
    OPP_PLAYER_STATS = ['DRtg', 'USG%', 'TOV%', 'BLK%', 'STL%', 'TRB%', 'ORB%', 
                        'DRB%', 'FTr', 'TS%', 'eFG%', 'ORtg', '3PAr', 
                        'StopPercent', 'FloorPercent']
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
                          'plus_minus_on', 'plus_minus_net', 'tov_bad_pass', 'tov_lost_ball', 'tov_other',
                          'fouls_shooting', 'fouls_blocking', 'fouls_offensive', 'fouls_take',
                          'and1s', 'astd_pts', 'fga_blkd', 'drawn_shooting']
    POS_ALLOWED_STATS = ['FG_per100poss', 'FG%', '3P_per100poss', '3P%', 'FTr', 'TRB%', 'AST%', 'STL%', 'BLK%', 
                         'TOV%', '+/-', 'TS%', 'eFG%', 'ORtg', 'DRtg', 'PTS/FGA', 'FTA/FGA']
    # taking the differential between game stats and season average
    TEAMSHARE_STATS = ['MP', 'FGA', '3PA', 'FTA', 'TRB', 'AST', 'STL', 'BLK', 'TOV']
    POSSESS_NORM_STATS = ["FT_per100poss", "3P_per100poss", "TOV_per100poss", "FG_per100poss", "3PA_per100poss", 
                          "DRB_per100poss", "AST_per100poss", "PF_per100poss", "PTS_per100poss", "FGA_per100poss", 
                          "STL_per100poss", "TRB_per100poss", "FTA_per100poss", "BLK_per100poss", "ORB_per100poss"]
    GAME_NORM_STATS = ['tov_bad_pass', 'tov_lost_ball', 'tov_other',
                       'fouls_shooting', 'fouls_blocking', 'fouls_offensive', 'fouls_take',
                       'and1s', 'astd_pts', 'fga_blkd', 'drawn_shooting']
    ONOFF_STATS = ['diff_tov_pct', 'diff_trb_pct', 'diff_ast_pct', 'diff_efg_pct', 
                   'diff_off_rtg', 'diff_blk_pct', 'diff_orb_pct', 'diff_drb_pct', 'diff_stl_pct']
    TWOMAN_STATS = ['diff_fta', 'diff_trb', 'diff_trb_pct', 'diff_pf', 'diff_blk', 'diff_fg_pct', 'diff_fg3_pct', 
                    'diff_ast', 'diff_drb', 'diff_stl', 'diff_fga', 'diff_efg_pct', 'diff_fg3a', 'diff_ft', 'diff_fg', 
                    'diff_pts', 'diff_orb_pct', 'diff_drb_pct', 'diff_fg3', 'diff_orb', 'diff_ft_pct', 'diff_tov']
    INTENSITY_STATS = ['own_win_rate','opp_win_rate','own_streak','opp_streak',
                       'own_aways_in_6','opp_aways_in_6','mp_in_6']


    ESPN_TRACKING_STATS = [] # TODO: CANT USE THESE UNLESS WE CAN GET THEM FOR MOST PLAYERS
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

            self.invalids.extend(self.target_game_info.get('inactive',[]))

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
        self.own_team_id = self.own_team['_id']
        self.opp_team_id = self.opp_team['_id']
        self.own_all_ids = self.own_team['all_ids']
        self.opp_all_ids = self.opp_team['all_ids']
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

    def averageStats(self, stats, allowed_keys, weights=None):
        if weights is not None and len(weights) != len(stats):
            raise Exception("Weights not same length as stats")
        trajectories = {k:[] for k in allowed_keys}
        for stat in stats:
            for k in allowed_keys:
                trajectories[k].append(stat.get(k))
        # average out the stats
        for k,v in trajectories.iteritems():
            filtered_values = []
            filtered_weights = []
            for i,value in enumerate(v):
                if value is not None and value != '':
                    new_weight = weights[i] if weights is not None else 1.0
                    filtered_values.append(value)
                    filtered_weights.append(new_weight)
            if len(filtered_values) > 0:
                trajectories[k] = np.average(filtered_values, weights=filtered_weights)
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
                if np.isnan(base_val) or np.isnan(reflect_val):
                    cur_diff[k] = 0.0
                elif isinstance(base_val,float) and isinstance(reflect_val,float):
                    cur_diff[k] = base_val - reflect_val
                else:
                    cur_diff[k] = 0.0
            differentials.append(cur_diff)
        return differentials

    def getStarters(self):
        '''
        Position choosing is kind of iffy

        Possible infinite loop with assigning starters to positions
        '''
        if self.target_game_info is not None:
            # find starters
            home_starters = self.target_game_info['home_starters']
            away_starters = self.target_game_info['away_starters']
            own_starters = home_starters if self.home_id in self.own_all_ids else away_starters
            opp_starters = home_starters if self.home_id in self.opp_all_ids else away_starters

            # find bench players
            opp_played = player_game_collection.find({"game_id" : self.target_game_info['_id'], "player_team" : {"$in" : self.opp_all_ids}})
            own_played = player_game_collection.find({"game_id" : self.target_game_info['_id'], "player_team" : {"$in" : self.own_all_ids}})
            opp_pid = [_['player_id'] for _ in opp_played if _['MP'] > 0.0]
            own_pid = [_['player_id'] for _ in own_played if _['MP'] > 0.0]

            # add players who didn't play to invalids (effectively invalid)
            self.invalids = list(set(self.invalids) | set([_['player_id'] for _ in opp_played if ['MP'] == 0.0]) | set([_['player_id'] for _ in own_played if ['MP'] == 0.0]))

            opp_bench = list(set(opp_pid) - set(opp_starters) - set(self.invalids))
            own_bench = list(set(own_pid) - set(own_starters) - set(self.invalids))
        else:
            # get espn depth chart
            charts = depth_collection.find({"time" : {"$gt" : self.season_start}})
            chosen_chart = sorted(charts, key=lambda x: abs(self.ts-x['time']))[0]
            chart_invalids = chosen_chart['invalids']
            chart_depth = chosen_chart['stats']
            own_bench = chart_depth[self.own_team_id]
            opp_bench = chart_depth[self.opp_team_id]
            
            # pick out starters
            own_starters = []
            opp_starters = []
            for k,v in own_bench.items():
                own_starters.append(v.pop(0))
            for k,v in opp_bench.items():
                opp_starters.append(v.pop(0))

            # flatten bench
            own_bench_flat = []
            opp_bench_flat = []
            for k,v in own_bench.items():
                own_bench_flat.extend(v)
            for k,v in opp_bench.items():
                opp_bench_flat.extend(v)
            own_bench = own_bench_flat
            opp_bench = opp_bench_flat

            # add invalids
            self.invalids.extend(chart_invalids)
        
        #print "Own: %s" % (own_starters+own_bench)
        #print "Opp: %s" % (opp_starters+opp_bench)


        # determine play distributions for both rosters
        own_distr = self.getPlayDistributions(own_starters + own_bench, self.own_all_ids)
        opp_distr = self.getPlayDistributions(opp_starters + opp_bench, self.opp_all_ids)
        self.pos_distr = own_distr['Pos']
        self.pos_distr.update(opp_distr['Pos'])
        self.play_distr = own_distr['TotPoss']
        self.play_distr.update(opp_distr['TotPoss'])
        self.usg_distr = own_distr['USG%']
        self.usg_distr.update(opp_distr['USG%'])
        self.floor_distr = own_distr['PProd']
        self.floor_distr.update(opp_distr['PProd'])
        self.stop_distr = own_distr['Stops']
        self.stop_distr.update(opp_distr['Stops'])

        self.own_play_distr_sum = sum([np.array(self.play_distr[_]) for _ in own_starters+own_bench])
        self.own_play_distr_sum /= sum(self.own_play_distr_sum)
        self.own_usg_distr_sum = sum([np.array(self.usg_distr[_]) for _ in own_starters+own_bench])
        self.own_usg_distr_sum /= sum(self.own_usg_distr_sum)
        self.own_floor_distr_sum = sum([np.array(self.floor_distr[_]) for _ in own_starters+own_bench])
        self.own_floor_distr_sum /= sum(self.own_floor_distr_sum)
        self.own_stop_distr_sum = sum([np.array(self.stop_distr[_]) for _ in own_starters+own_bench])
        self.own_stop_distr_sum /= sum(self.own_stop_distr_sum)
        self.opp_play_distr_sum = sum([np.array(self.play_distr[_]) for _ in opp_starters+opp_bench])
        self.opp_play_distr_sum /= sum(self.opp_play_distr_sum)
        self.opp_usg_distr_sum = sum([np.array(self.usg_distr[_]) for _ in opp_starters+opp_bench])
        self.opp_usg_distr_sum /= sum(self.opp_usg_distr_sum)
        self.opp_floor_distr_sum = sum([np.array(self.floor_distr[_]) for _ in opp_starters+opp_bench])
        self.opp_floor_distr_sum /= sum(self.opp_floor_distr_sum)
        self.opp_stop_distr_sum = sum([np.array(self.stop_distr[_]) for _ in opp_starters+opp_bench])
        self.opp_stop_distr_sum /= sum(self.opp_stop_distr_sum)



        '''
        print self.own_team_id
        print "PLAYS: %s" % self.own_play_distr_sum
        print "USG: %s" % self.own_usg_distr_sum
        print "PPROD: %s" % self.own_floor_distr_sum
        print "STOPS: %s" % self.own_stop_distr_sum
        print self.opp_team_id
        print "PLAYS: %s" % self.opp_play_distr_sum
        print "USG: %s" % self.opp_usg_distr_sum
        print "PPROD: %s" % self.opp_floor_distr_sum
        print "STOPS: %s" % self.opp_stop_distr_sum

        # test graph
        fig1 = plt.figure()
        p1 = plt.plot(self.own_play_distr_sum, marker='o', label='plays')
        p2 = plt.plot(self.own_usg_distr_sum, marker='o', label='usg')
        p3 = plt.plot(self.own_floor_distr_sum, marker='o', label='pprod')
        p4 = plt.plot(self.own_stop_distr_sum, marker='o', label='stops')
        fig1.suptitle(self.own_team_id, fontsize=20)
        p_labels = plt.xticks(range(5), self.POSITIONS, rotation='vertical')
        plt.legend(loc='best', shadow=True)

        fig2 = plt.figure()
        p1 = plt.plot(self.opp_play_distr_sum, marker='o', label='plays')
        p2 = plt.plot(self.opp_usg_distr_sum, marker='o', label='usg')
        p3 = plt.plot(self.opp_floor_distr_sum, marker='o', label='pprod')
        p4 = plt.plot(self.opp_stop_distr_sum, marker='o', label='stops')
        fig2.suptitle(self.opp_team_id, fontsize=20)
        p_labels = plt.xticks(range(5), self.POSITIONS, rotation='vertical')
        plt.legend(loc='best', shadow=True)

        plt.show(block=True)
        '''

        # label the values
        self.own_play_distr_sum = dict(zip(self.POSITIONS, self.own_play_distr_sum))
        self.own_usg_distr_sum = dict(zip(self.POSITIONS, self.own_usg_distr_sum))
        self.own_floor_distr_sum = dict(zip(self.POSITIONS, self.own_floor_distr_sum))
        self.own_stop_distr_sum = dict(zip(self.POSITIONS, self.own_stop_distr_sum))
        self.opp_play_distr_sum = dict(zip(self.POSITIONS, self.opp_play_distr_sum))
        self.opp_usg_distr_sum = dict(zip(self.POSITIONS, self.opp_usg_distr_sum))
        self.opp_floor_distr_sum = dict(zip(self.POSITIONS, self.opp_floor_distr_sum))
        self.opp_stop_distr_sum = dict(zip(self.POSITIONS, self.opp_stop_distr_sum))


        # switch out invalid starters for bench player if needed
        for k in opp_starters:
            if k in self.invalids:
                # get position
                rankings = {}
                for pid in opp_bench:
                    rankings[pid] = distance.cityblock(self.play_distr[k], self.play_distr[pid])
                ranked = sorted(rankings.items(), key=lambda x: x[1])
                for chosen, pid_dist in ranked:
                    if chosen not in opp_starters:
                        opp_starters.remove(k)
                        opp_starters.append(chosen)
                        break
        for k in own_starters:
            if k in self.invalids:
                # get position
                rankings = {}
                for pid in own_bench:
                    rankings[pid] = distance.cityblock(self.play_distr[k], self.play_distr[pid])
                ranked = sorted(rankings.items(), key=lambda x: x[1])
                for chosen, pid_dist in ranked:
                    if chosen not in own_starters:
                        own_starters.remove(k)
                        own_starters.append(chosen)
                        break

        '''
        print "Own Starters: %s" % own_starters
        print "Own Bench: %s" % own_bench
        print "Opp Starters: %s" % opp_starters
        print "Opp Bench: %s" % opp_bench
        '''

        self.own_starters = own_starters
        self.own_bench = own_bench
        self.opp_starters = opp_starters
        self.opp_bench = opp_bench

        if self.player_id not in self.own_starters + self.own_bench:
            raise Exception("Player not playing in game...")

        # find player row
        self.player_row = player_collection.find_one({"_id" : self.player_id})

        # find player starting or not
        self.starting = self.CAT_ONE_HOT_ON if self.player_id in self.own_starters else self.CAT_ONE_HOT_OFF


    def getAdvancedStatsForPlayer(self, player_id, team_ids, filter_keys):
        '''
        TODO CHANGE THIS TO ESPN ADVANCED STATS
        '''
        advanced_rows = list(advanced_collection.find({"player_id" : player_id, "team_id" : {"$in" : team_ids}, "time" : {"$gt" : self.season_start, "$lt" : self.season_start + timedelta(365)}}))
        if len(advanced_rows) == 0:
            raise Exception("Could not find valid advanced stats row for %s, %s, %s" % (player_id, team_ids, self.season_start))
        chosen_advanced_row = sorted(advanced_rows, key=lambda x: abs(self.ts - x['time']))[0]
        # normalize by 48 minutes for the specified keys
        mp = float(chosen_advanced_row["MP"])
        by36 = mp/36.0
        for k in self.GAME_NORM_STATS:
            if k in chosen_advanced_row:
                stat_value = chosen_advanced_row.get(k,0.0)
                if stat_value is None:
                    stat_value = 0.0
                normed = stat_value / by36
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
        player_stats = player_game_collection.find({"player_id": self.player_id, "player_team" : {"$in" : self.own_all_ids}, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, sort=[("game_time",-1)])
        player_stats = self.calculateAdditionalPlayerStats(player_stats)

        # check enough player stats
        if len(player_stats) < 2:
            raise Exception("Not enough previous games for player %s" % self.player_id)

        # get player stats running stat avgs and variances
        player_last_matchups = [_ for _ in player_stats if _['game_id'] in self.matchup_gids]
        self.player_last_matchup_avgs = self.averageStats(player_last_matchups, self.PLAYER_STATS)
        self.player_season_avg = self.averageStats(player_stats, self.PLAYER_STATS)
        self.player_running_avg = self.averageStats(player_stats[:10], self.PLAYER_STATS)
        self.player_running_variance = self.varianceStats(player_stats[:10], self.PLAYER_STATS)

        height = self.player_row['height']
        weight = self.player_row['weight']
        shoots = self.CAT_ONE_HOT_ON if self.player_row.get('shoots','R') == 'R' else self.CAT_ONE_HOT_OFF
        self.physical_stats = {'height' : height,
                               'weight' : weight,
                               'shoots' : shoots}


        '''
        # TODO CHANGE THIS TO NEW ESPN ADVANCED STATS
        espn_rows = espn_player_stat_collection.find({"player_id" : self.player_id, "time" : {"$gt" : self.season_start}})
        chosen_espn_row = sorted(espn_rows, key=lambda x: abs(self.ts - x['time']))[0]
        self.player_espn_stats = {k: chosen_espn_row[k] for k in self.ESPN_TRACKING_STATS}
        '''

        # get advanced stats
        self.player_advanced_stats = self.getAdvancedStatsForPlayer(self.player_id, self.own_all_ids, self.ADVANCED_STATS)

        # get team effects (stat fractions)
        # MP, FGA, 3PA, FTA, TRB, AST, STL, BLK, TOV
        teamshare_stats = defaultdict(list)
        for g in self.own_last_10_games:
            gid = g['_id']
            g_players = list(player_game_collection.find({"game_id": gid, "player_team": {"$in" : self.own_all_ids}}))
            p_row = player_game_collection.find_one({"game_id": gid, "player_id": self.player_id})
            if p_row is None:
                p_row = {}
            for s in self.TEAMSHARE_STATS:
                total = sum([float(x.get(s,0.0)) for x in g_players])
                if total > 0.0:
                    fraction = float(p_row.get(s,0.0)) / total
                    teamshare_stats[s].append(fraction)
        self.teamshare = {k: np.mean(v) for k,v in teamshare_stats.items()}

        # get possible matchup player production
        facing_player_rankings = self.rankMatchupPossibility()
        self.pos_starter = facing_player_rankings[0]
        self.pos_backup = facing_player_rankings[1]

        starter_row = player_collection.find_one({"_id" : self.pos_starter})
        backup_row = player_collection.find_one({"_id" : self.pos_backup})
        starter_stats = player_game_collection.find({"player_id" : self.pos_starter, "game_time" : {"$lt" : self.ts, "$gt" : self.season_start}},
                                                    sort=[("game_time",-1)], limit=10)
        starter_stats = self.calculateAdditionalPlayerStats(starter_stats)
        backup_stats = player_game_collection.find({"player_id" : self.pos_backup, "game_time" : {"$lt" : self.ts, "$gt" : self.season_start}},
                                                    sort=[("game_time",-1)], limit=10)
        backup_stats = self.calculateAdditionalPlayerStats(backup_stats)
        
        if len(starter_stats) < 1:
            print "No last games found for %s" % self.pos_starter
            self.opp_starter_avgs = {_: np.nan for _ in self.OPP_PLAYER_STATS}
        else:
            self.opp_starter_avgs = self.averageStats(starter_stats, self.OPP_PLAYER_STATS)
        if len(backup_stats) < 1:
            print "No last games found for %s" % self.pos_backup
            self.opp_backup_avgs = {_: np.nan for _ in self.OPP_PLAYER_STATS}
        else:
            self.opp_backup_avgs = self.averageStats(backup_stats, self.OPP_PLAYER_STATS)

        try:
            self.pos_starter_advanced_stats = self.getAdvancedStatsForPlayer(self.pos_starter, self.opp_all_ids, self.OPP_ADVANCED_STATS)
        except Exception as e:
            print "No advanced stats found for %s" % self.pos_starter
            self.pos_starter_advanced_stats = {_: np.nan for _ in self.OPP_ADVANCED_STATS}
        try:
            self.pos_backup_advanced_stats = self.getAdvancedStatsForPlayer(self.pos_backup, self.opp_all_ids, self.OPP_ADVANCED_STATS)
        except Exception as se:
            print "No advanced stats found for %s" % self.pos_backup
            self.pos_backup_advanced_stats = {_: np.nan for _ in self.OPP_ADVANCED_STATS}

        self.opp_starter_avgs['height_difference'] = height - starter_row['height']
        self.opp_starter_avgs['weight_difference'] = weight - starter_row['weight']
        self.opp_backup_avgs['height_difference'] = height - backup_row['height']
        self.opp_backup_avgs['weight_difference'] = weight - backup_row['weight']

        # calculate position earned/givenup
        self.calculatePositionStats(self.opp_last_10_games)

    def getPlayDistributions(self, pids, team_ids, stat_keys=None):
        if stat_keys is None:
            stat_keys = ['Pos','TotPoss', 'USG%', 'PProd', 'Stops']

        # get the position estimates for each player
        pos_estimates = {pid: self.getPositionEstimateForPlayer(pid, team_ids) for pid in pids}

        last_games = list(team_game_collection.find({"team_id": {"$in" : team_ids}, "game_time": {"$lt": self.ts, "$gt": self.ts - timedelta(days=60)}}, 
                                                    sort=[("game_time",-1)]))
        last_gids = [_['game_id'] for _ in last_games[:10]]
        if len(last_gids) == 0:
            raise Exception("Not enough games found for %s at %s" % (team_ids,self.ts))

        modulated = {_: {} for _ in stat_keys}
        for pid in pids:
            sums = {_: 0.0 for _ in stat_keys}
            for gid in last_gids:
                pid_row = player_game_collection.find_one({"player_id" : pid, "game_id": gid})
                if pid_row and pid_row['MP'] > 0.0:
                    for k in stat_keys:
                        sums[k] += pid_row.get(k, 0.0)
            for k in stat_keys:
                if k == 'Pos':
                    stat_mod = 1.0
                else:
                    stat_mod = sums[k] / len(last_gids)
                modulated[k][pid] = np.array(pos_estimates[pid]) * stat_mod

        return modulated

    def rankMatchupPossibility(self):
        '''
        To find facing starter and backup production, find the players who play the the most possessions, 
        as well as the positions they play, then find ones that are most statistically likely matchups 
        by poss frequency and pos percentage

        if person is starting, make sure a starter is matched up first
        '''
        # recreate opponent roster
        all_players = self.opp_starters + self.opp_bench

        # multiply percentages by total plays, and total overlapping plays
        # rank by distance metric to own play distribution
        ranks = {}
        own_plays = sum(self.play_distr[self.player_id]) / (100.0 * sum(self.pos_distr[self.player_id]))
        base_vector = np.concatenate([self.pos_distr[self.player_id],[own_plays]])
        #print "BASE: %s" % base_vector
        for pid in all_players:
            # don't include non-starters without any plays
            '''
            if pid in self.opp_bench and sum(self.play_distr[pid]) == 0.0:
                print "ignoring %s" % pid
                continue
            '''
            plays = sum(self.play_distr[pid]) / (100.0 * sum(self.pos_distr[pid]))
            other_vector = np.concatenate([self.pos_distr[pid],[plays]])
            dist = distance.cityblock(base_vector, other_vector)
            #print "%s: %s, %s" % (pid, other_vector, dist)

            ranks[pid] = dist

        # sort
        ranked_players = sorted(ranks.items(), key=lambda x: x[1])
        ranked_pids = [_[0] for _ in ranked_players]

        if self.starting == self.CAT_ONE_HOT_ON:
            # pull a starter up to first
            for i, pid in enumerate(ranked_pids):
                if pid in self.opp_starters:
                    #print "Pulling %s up front" % pid
                    # push i up to 0
                    new_first = ranked_pids.pop(i)
                    ranked_pids.insert(0, new_first)
                    break

        #print ranked_players

        return ranked_pids


    def getPositionEstimateForPlayer(self, pid, team_ids):
        # try to get advanced row
        try:
            p_advanced = self.getAdvancedStatsForPlayer(pid, team_ids, self.ADVANCED_STATS)
        except Exception as e:
            traceback.print_exc()
            p_advanced = None
        if p_advanced:
            # derive position percentages
            pos_percents = [float(p_advanced["pct_%s" % k.lower()]) for k in self.POSITIONS]
        else:
            # use listed positions instead
            p_basic = player_collection.find_one({"_id" : pid})
            if p_basic is None:
                raise Exception("advanced row and player row both not found")
            raw_weights = [1.0/(float(i+1)**2) for i in range(len(p_basic['position']))]
            pos_weights = list(np.array(raw_weights) / np.sum(raw_weights))
            ref_dict = dict(zip(p_basic['position'], pos_weights))
            pos_percents = []
            for p in self.POSITIONS:
                if p in ref_dict:
                    pos_percents.append(ref_dict[p])
                else:
                    pos_percents.append(0.0)
        return pos_percents

    def calculatePositionStats(self, opp_last_10_games):
        '''
        find earned/given up stats per position, by differential between game stats and season average

        Instead of allowed stats for each position, 
        provide allowed stats by taking position percent estimates into account.
        
        Taking each training point, and adding to each positions allowed average by the players position, 
        and then mix the distributions by the target player's pos percents.
        '''
        season_stats = {}
        player_blacklist = set()
        earned_position_stats = {p:[] for p in self.POSITIONS}
        givenup_position_stats = {p:[] for p in self.POSITIONS}
        for game_row in opp_last_10_games:
            game_id = game_row['_id']
            opp_opp_id = game_row['home_id'] if game_row['home_id'] not in self.opp_all_ids else game_row['away_id']
            opp_opp_all_ids = team_collection.find_one({"all_ids": opp_opp_id})['all_ids']
            earned_players = player_game_collection.find({"game_id" : game_id, "player_team": {"$in" : self.opp_all_ids}})
            givenup_players = player_game_collection.find({"game_id" : game_id, "player_team": {"$in" : opp_opp_all_ids}})
            earned_players = self.calculateAdditionalPlayerStats(earned_players)
            givenup_players = self.calculateAdditionalPlayerStats(givenup_players)
            for ep in earned_players:
                pid = ep['player_id']
                # try to get advanced row
                try:
                    p_advanced = self.getAdvancedStatsForPlayer(pid, self.opp_all_ids, self.ADVANCED_STATS)
                except Exception as e:
                    print e
                    p_advanced = None
                if p_advanced:
                    # derive position percentages
                    for pos in self.POSITIONS:
                        key = "pct_%s" % pos.lower()
                        pos_percent = float(p_advanced[key])
                        if pos_percent > 0.0:
                            earned_position_stats[pos].append((pos_percent, ep))
                else:
                    # use listed positions instead
                    p_basic = player_collection.find_one({"_id" : pid})
                    if p_basic is None:
                        continue
                    raw_weights = [1.0/(float(i+1)**2) for i in range(len(p_basic['position']))]
                    pos_weights = list(np.array(raw_weights) / np.sum(raw_weights))
                    for pos, pos_percent in zip(p_basic['position'], pos_weights):
                        earned_position_stats[pos].append((pos_percent, ep))
                # check for season stats (60 day window avg)
                if pid not in season_stats and pid not in player_blacklist:
                    player_team = ep['player_team']
                    pid_season = player_game_collection.find({"player_team" : player_team, "player_id" : pid, "game_time" : {"$gt" : self.ts - timedelta(days=30), "$lt" : self.ts + timedelta(days=30)}})
                    pid_season = self.calculateAdditionalPlayerStats(pid_season)
                    if len(pid_season) < 3:
                        player_blacklist.add(pid)
                    else:
                        season_stats[pid] = self.averageStats(pid_season, self.POS_ALLOWED_STATS)
            for gp in givenup_players:
                pid = gp['player_id']
                # try to get advanced row
                try:
                    p_advanced = self.getAdvancedStatsForPlayer(pid, opp_opp_all_ids, self.ADVANCED_STATS)
                except Exception as e:
                    print e
                    p_advanced = None
                if p_advanced:
                    # derive position percentages
                    for pos in self.POSITIONS:
                        key = "pct_%s" % pos.lower()
                        pos_percent = float(p_advanced[key])
                        if pos_percent > 0.0:
                            givenup_position_stats[pos].append((pos_percent, gp))
                else:
                    # use listed positions instead
                    p_basic = player_collection.find_one({"_id" : pid})
                    if p_basic is None:
                        continue
                    raw_weights = [1.0/(float(i+1)**2) for i in range(len(p_basic['position']))]
                    pos_weights = list(np.array(raw_weights) / np.sum(raw_weights))
                    for pos, pos_percent in zip(p_basic['position'], pos_weights):
                        givenup_position_stats[pos].append((pos_percent, gp))
                # check for season stats
                if pid not in season_stats and pid not in player_blacklist:
                    player_team = gp['player_team']
                    pid_season = player_game_collection.find({"player_team" : player_team, "player_id" : pid, "game_time" : {"$gt" : self.ts - timedelta(days=30), "$lt" : self.ts + timedelta(days=30)}})
                    pid_season = self.calculateAdditionalPlayerStats(pid_season)
                    if len(pid_season) < 3:
                        player_blacklist.add(pid)
                    else:
                        season_stats[pid] = self.averageStats(pid_season, self.POS_ALLOWED_STATS)

        # average earned/givenup for each position by their weights, ratio-ed with the season stats
        for k in earned_position_stats.keys():
            v = earned_position_stats[k]
            weights = []
            stats = []
            for w, p_row in v:
                if p_row['player_id'] in player_blacklist:
                    continue
                elif p_row['MP'] < 10.0:
                    continue
                pid_season_avg = season_stats[p_row['player_id']]
                ratios = {}
                for statkey in self.POS_ALLOWED_STATS:
                    num = p_row[statkey]
                    denum = pid_season_avg[statkey]
                    try:
                        num = float(num)
                        denum = float(denum)
                    except Exception as e:
                        ratios[k] = 1.0
                        continue
                    ratios[statkey] = num/denum if denum > 0.0 else 1.0  
                stats.append(ratios)
                weights.append(w)
            avg_stats = self.averageStats(stats, self.POS_ALLOWED_STATS, weights=weights)
            earned_position_stats[k] = avg_stats
        for k in givenup_position_stats.keys():
            v = givenup_position_stats[k]
            weights = []
            stats = []
            for w, p_row in v:
                if p_row['player_id'] in player_blacklist:
                    continue
                elif p_row['MP'] < 10.0:
                    continue
                pid_season_avg = season_stats[p_row['player_id']]
                ratios = {}
                for statkey in self.POS_ALLOWED_STATS:
                    num = p_row[statkey]
                    denum = pid_season_avg[statkey]
                    try:
                        num = float(num)
                        denum = float(denum)
                    except Exception as e:
                        ratios[k] = 1.0
                        continue
                    ratios[statkey] = num/denum if denum > 0.0 else 1.0  
                stats.append(ratios)
                weights.append(w)
            avg_stats = self.averageStats(stats, self.POS_ALLOWED_STATS, weights=weights)
            givenup_position_stats[k] = avg_stats

        # find base player position estimates
        if self.player_advanced_stats:
            pos_estimates = [float(self.player_advanced_stats['pct_%s' % p.lower()]) for p in self.POSITIONS]
        else:
            raw_weights = [1.0/(float(i+1)**2) for i in range(len(self.player_row['position']))]
            pos_weights = list(np.array(raw_weights) / np.sum(raw_weights))
            weight_dict = {pos : percent for pos,percent in zip(self.player_row['position'],pos_weights)}
            pos_estimates = [weight_dict[k] if k in weight_dict else 0.0 for k in self.POSITIONS]
        
        # use base player position estimates to blend givenup/earned stats together
        givenup_stats = []
        earned_stats = []
        weights = []
        for pos, est in zip(self.POSITIONS, pos_estimates):
            if est > 0.0:
                givenup_stats.append(givenup_position_stats[pos])
                earned_stats.append(earned_position_stats[pos])
                weights.append(est)
        self.opp_pos_givenup = self.averageStats(givenup_stats, self.POS_ALLOWED_STATS, weights=weights)
        self.opp_pos_earned = self.averageStats(earned_stats, self.POS_ALLOWED_STATS, weights=weights)

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
        own_results = list(team_game_collection.find({"team_id": {"$in" : self.own_all_ids}, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, 
                                            sort=[("game_time",-1)]))
        if len(own_results) < 2:
            raise Exception("Could not find more than 3 prior games for team: %s" % self.own_all_ids)
        own_reflection_results = [team_game_collection.find_one({"game_id": g['game_id'], "team_id": {"$nin": self.own_all_ids}}) for g in own_results]
        own_results = self.calculateAdditionalTeamStats(own_results, own_reflection_results)
        own_matchup_results = [own for own,refl in zip(own_results,own_reflection_results) if refl['team_id'] in self.opp_all_ids]

        # same gids for later use
        self.matchup_gids = [_['game_id'] for _ in own_matchup_results]

        self.own_most_recent_game = game_collection.find_one({"_id" : own_results[0]['game_id']})
        self.own_last_10 = copy.deepcopy(own_results[:10])
        self.own_last_10_games = [game_collection.find_one({"_id": g['game_id']}) for g in self.own_last_10]
        self.own_gametimes = [o['game_time'] for o in own_results[:5]]
        self.own_team_season_avgs = self.averageStats(own_results, self.TEAM_STATS)
        self.own_team_running_avgs = self.averageStats(own_results[:10], self.TEAM_STATS)
        self.own_team_matchup_avgs = self.averageStats(own_matchup_results[:10], self.TEAM_STATS)

        #self.own_reflection_season_avgs = self.averageStats(own_reflection_results, self.TEAM_STATS)
        #self.own_reflection_running_avgs = self.averageStats(own_reflection_results[:10], self.TEAM_STATS)

        # opp season/running avg team stats ( + reflection)
        opp_results = list(team_game_collection.find({"team_id": {"$in" : self.opp_all_ids}, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, 
                                            sort=[("game_time",-1)]))
        if len(opp_results) < 2:
            raise Exception("Could not find more than 3 prior games for team: %s" % self.opp_all_ids)

        opp_reflection_results = [team_game_collection.find_one({"game_id": g['game_id'], "team_id": {"$nin": self.opp_all_ids}}) for g in opp_results]
       
        opp_results = self.calculateAdditionalTeamStats(opp_results, opp_reflection_results)
        opp_matchup_results = [opp for opp,refl in zip(opp_results,opp_reflection_results) if refl['team_id'] in self.own_all_ids]

        self.opp_most_recent_game = game_collection.find_one({"_id" : opp_results[0]['game_id']})
        self.opp_last_10 = copy.deepcopy(opp_results[:10])
        self.opp_last_10_games = [game_collection.find_one({"_id": g['game_id']}) for g in self.opp_last_10]
        self.opp_gametimes = [o['game_time'] for o in opp_results[:5]]
        self.opp_team_season_avgs = self.averageStats(opp_results, self.TEAM_STATS)
        self.opp_team_running_avgs = self.averageStats(opp_results[:10], self.TEAM_STATS)
        self.opp_team_matchup_avgs = self.averageStats(opp_matchup_results[:10], self.TEAM_STATS)
        #self.opp_reflection_season_avgs = self.averageStats(opp_reflection_results, self.TEAM_STATS)
        #self.opp_reflection_running_avgs = self.averageStats(opp_reflection_results[:10], self.TEAM_STATS)

    def calculateAdditionalPlayerStats(self, input_rows):
        '''
        Calculating
            PTS/FGA = PTS / FGA
            FTA/FGA = FTA / FGA
        '''
        results = []
        for row in input_rows:
            new_row = copy.deepcopy(row)
            if new_row['MP'] > 0.0:
                new_row['PTS/FGA'] = float(new_row['PTS']) / float(new_row['FGA']) if float(new_row['FGA']) > 0.0 else None
                new_row['FTA/FGA'] = float(new_row['FTA']) / float(new_row['FGA']) if float(new_row['FGA']) > 0.0 else None
            results.append(new_row)
        return results

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
            opp_2PA
            opp_3PA
            opp_2P%
            opp_3P%"
        '''
        if len(own_results) != len(opp_results):
            raise Exception("own team results and opp team results not same length")
        results = []
        for own,opp in zip(own_results, opp_results):
            new_row = copy.deepcopy(own)
            new_row['opp_eFG%'] = float(opp['eFG%'])
            new_row['opp_TOV%'] = float(opp['TOV%'])
            new_row['opp_FT/FGA'] = float(opp['FT/FGA'])
            new_row['opp_2PA'] = float(opp['2PA'])
            new_row['opp_3PA'] = float(opp['3PA'])
            new_row['opp_2P%'] = float(opp['2P%'])
            new_row['opp_3P%'] = float(opp['3P%'])
            new_row['DRB%'] = 100.0 - float(opp['ORB%'])
            new_row['DRtg'] = float(opp['ORtg'])
            new_row['NetRtg'] = float(own['ORtg']) - float(opp['ORtg'])
            new_row['T_net'] = float(own['T']) - float(opp['T'])
            results.append(new_row)
        return results

    def getEffectStats(self):
        '''
        team effects: 
            average player on/off stats for injured players
        player effects:
            look at 2-man combinations with the base player
                - take the player average effects
                - average the 2-man combination effects with the players that are going to be out
                - take the difference between (2) and (1) - that gives you the additional synergistic effects that are going to be lost
        '''
        #print "INVALIDS: %s" % self.invalids

        try:
            self.player_onoff = self.getOnOffStatsForPlayer(self.player_id, self.own_team_id, self.ONOFF_STATS)
        except Exception as e:
            self.invalid_onoffs = {k : 0.0 for k in self.ONOFF_STATS}
            self.player_onoff = {k : 0.0 for k in self.ONOFF_STATS}

        invalid_onoff_rows = []
        for pid in self.invalids:
            try:
                invalid_onoff = self.getOnOffStatsForPlayer(pid, self.own_team_id, self.ONOFF_STATS)
                invalid_onoff_rows.append(invalid_onoff)
            except Exception as e:
                #print e
                continue
        if len(invalid_onoff_rows) > 0:
            self.invalid_onoffs = self.averageStats(invalid_onoff_rows, self.ONOFF_STATS)
        else:
            self.invalid_onoffs = {k : 0.0 for k in self.ONOFF_STATS}


        try:
            self.player_twoman = self.getTwoManStatsForPlayers([self.player_id], self.own_team_id, self.TWOMAN_STATS)
            invalid_twomans = []
            for pid in self.invalids:
                try:
                    invalid_twoman = self.getTwoManStatsForPlayers([self.player_id, pid], self.own_team_id, self.TWOMAN_STATS)
                    invalid_twomans.append(invalid_twoman)
                except Exception as e:
                    #print e
                    continue
            if len(invalid_twomans) > 0:
                twoman_diffs = self.takeDifference(invalid_twomans, [self.player_twoman] * len(invalid_twomans), self.TWOMAN_STATS)
                self.invalid_twomans = self.averageStats(twoman_diffs, self.TWOMAN_STATS)
            else:
                self.invalid_twomans = {k : 0.0 for k in self.TWOMAN_STATS}
        except Exception as e:
            self.player_twoman = {k : 0.0 for k in self.TWOMAN_STATS}
            self.invalid_twomans = {k : 0.0 for k in self.TWOMAN_STATS}

    def getOnOffStatsForPlayer(self, player_id, team_id, filter_keys):
        end_year = self.season_start.year + 1
        onoff_rows = list(onoff_collection.find({"player_id" : player_id, "team_id" : team_id, "year" : end_year}))
        if len(onoff_rows) == 0:
            raise Exception("Could not find valid onoff stats row for %s, %s, %s" % (player_id, team_id, end_year))
        chosen_onoff_row = sorted(onoff_rows, key=lambda x: abs(self.ts - x['time']))[0]
        player_onoff_stats = {k: chosen_onoff_row[k] for k in filter_keys}
        return player_onoff_stats

    def getTwoManStatsForPlayers(self, player_ids, team_id, filter_keys):
        end_year = self.season_start.year + 1
        chosen_row = None
        player_ids = sorted(player_ids)
        if len(player_ids) == 1:
            twoman_rows = list(two_man_collection.find({"player_one" : player_ids[0], "player_two" : None, "year" : end_year}))
        elif len(player_ids) == 2:
            twoman_rows = list(two_man_collection.find({"player_one" : player_ids[0], "player_two" : player_ids[1], "team_id" : team_id, "year" : end_year}))
        else:
            raise Exception("Invalid two man stat players input")

        for row in sorted(twoman_rows, key=lambda x: abs(self.ts - x['time'])):
            chosen_row = row
        if chosen_row is None:
            raise Exception("Could not find valid twoman stats row for %s, %s, %s" % (player_ids, team_id, end_year))

        # filter
        player_twoman_stats = {k: chosen_row[k] for k in filter_keys}
        return player_twoman_stats

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
        self.getEffectStats()

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
                     #'opp_starter' : copy.deepcopy(self.opp_starter_avgs),
                     #'opp_backup' : copy.deepcopy(self.opp_backup_avgs),
                     #'opp_starter_advanced': copy.deepcopy(self.pos_starter_advanced_stats),
                     #'opp_backup_advanced': copy.deepcopy(self.pos_backup_advanced_stats),
                     'own' : copy.deepcopy(self.player_running_avg),
                     'opp_pos_givenup': copy.deepcopy(self.opp_pos_givenup),
                     #'opp_pos_earned': copy.deepcopy(self.opp_pos_earned),
                     'net_indv': copy.deepcopy(self.player_twoman),
                     'net_team': copy.deepcopy(self.player_onoff)
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
        data = {#'own_espn_team_stats': copy.deepcopy(self.own_espn_team_stats),
                #'opp_espn_team_stats': copy.deepcopy(self.opp_espn_team_stats),
                'own_team_running_avgs': copy.deepcopy(self.own_team_running_avgs),
                'opp_team_running_avgs': copy.deepcopy(self.opp_team_running_avgs),
                'own_play_distr': copy.deepcopy(self.own_play_distr_sum),
                'own_usg_distr': copy.deepcopy(self.own_usg_distr_sum),
                'own_pprod_distr': copy.deepcopy(self.own_floor_distr_sum),
                'own_stop_distr': copy.deepcopy(self.own_stop_distr_sum),
                'opp_play_distr': copy.deepcopy(self.opp_play_distr_sum),
                'opp_usg_distr': copy.deepcopy(self.opp_usg_distr_sum),
                'opp_pprod_distr': copy.deepcopy(self.opp_floor_distr_sum),
                'opp_stop_distr': copy.deepcopy(self.opp_stop_distr_sum),
                }

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
        own_matchup_trend_diff = self.takeDifference([self.own_team_season_avgs], [self.own_team_matchup_avgs], self.TEAM_STATS)[0]
        opp_matchup_trend_diff = self.takeDifference([self.opp_team_season_avgs], [self.opp_team_matchup_avgs], self.TEAM_STATS)[0]
        player_matchup_diff = self.takeDifference([self.player_season_avg], [self.player_last_matchup_avgs], self.PLAYER_STATS)[0]

        data = {'own_recent_trend': own_trend_diff,
                'opp_recent_trend': opp_trend_diff,
                'own_matchup_trend': own_matchup_trend_diff,
                'opp_matchup_trend': opp_matchup_trend_diff,
                'player_matchup_trend': player_matchup_diff,
                #'invalid_onoffs': copy.deepcopy(self.invalid_onoffs),
                #'invalid_twomans': copy.deepcopy(self.invalid_twomans)
                }

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
        data['own_games_in_5_days'] = len([_ for _ in own_game_deltas if _ <= 5])
        data['opp_games_in_5_days'] = len([_ for _ in opp_game_deltas if _ <= 5])

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
        cont_labels = ['own_days_since_last', 'opp_days_since_last', 'own_games_in_5_days', 'opp_games_in_5_days']
        #cat_labels = ['team_%s' % k for k in self.all_teams] + ['location_%s' % k for k in self.all_locations]
        #cat_feature_splits = [len(self.all_teams), len(self.all_locations)]
        cat_labels = ['location_%s' % k for k in self.all_locations]
        cat_feature_splits = [len(self.all_locations)]
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

def dumpCSV(filepath, pid=None, limit=None, time=None, min_count=None, save=True):
    if save:
        # test the filepath is nonexistent
        if os.path.isfile(filepath):
            raise Exception("Manually delete old file")
        # open the file
        out_file = open(filepath, 'wb')
        writer = csv.writer(out_file)
    args = findAllTrainingGames(pid=pid, limit=limit, time=time, min_count=min_count)
    keys = None
    count = 0

    # multiprocess arguments
    pool = mp.Pool(5)
    results = pool.imap_unordered(parseFeaturesForCSV, args)

    # write results
    if save:
        for result in results:
            if result is not None:
                labels = result['labels']
                features = result['sorted_features']
                if count == 0:
                    writer.writerow(labels)
                writer.writerow(features)
                count += 1

def parseFeaturesForCSV(arg):
    id_keys = ['player_id', 'target_game']
    print arg
    try:
        fe = NBAFeatureExtractor(arg['player_id'], arg['team_id'], target_game=arg['target_game'])
        cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = fe.runEncoderFeatures()
        # aggregate keys and encode cat splits into key name
        all_dict = dict(zip(cont_labels, cont_features))
        new_cat_labels = []
        for i, split in enumerate(cat_feat_splits):
            split_keys = ["%s#%s" % (i, _) for _ in cat_labels[:split]]
            if len(split_keys) != split:
                raise Exception("Cat label splits are incorrect!")
            cat_labels = cat_labels[split:]
            new_cat_labels.extend(split_keys)
        all_dict.update(dict(zip(new_cat_labels, cat_features)))
        
        # aggregate labels and sort features
        keys = cont_labels + new_cat_labels
        labels = ["%s#%s" % ('id',_) for _ in id_keys] + keys
        sorted_features = [arg[_] for _ in id_keys]+[all_dict[_] for _ in keys]

        # return
        return {'labels' : labels,
                'sorted_features': sorted_features}
    except Exception as e:
        traceback.print_exc()
        return None

def parseFeaturesForUse(arg):
    try:
        fe = NBAFeatureExtractor(arg['player_id'], arg['team_id'], target_game=arg['target_game'])
        cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = fe.runEncoderFeatures()
        return (arg, cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)
    except Exception as e:
        traceback.print_exc()
        return None


def streamPlayerInput(pid, limit=None, time=None, min_count=None):
    args = findAllTrainingGames(pid=pid, limit=limit, time=time, min_count=min_count)
    keys = None
    count = 0

    # multiprocess arguments
    pool = mp.Pool(5)
    results = pool.imap_unordered(parseFeaturesForUse, args)

    for result in results:
        if result is not None:
            yield result
            count += 1

def streamCSV(filepath):
    # test the filepath is existent
    if not os.path.isfile(filepath):
        raise Exception("File does not exist")
    # open the file
    in_file = open(filepath, 'rb')
    reader = csv.reader(in_file)
    id_regex = re.compile("id#")
    cat_regex = re.compile("[0-9]+#")
    count = 0
    # cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits
    labels = None
    cat_feat_splits = []
    for row in reader:
        if count == 0:
            labels = row
            count += 1
            # find cat_feat_splits
            cat_split_counter = 0
            while True:
                prefix = "%s#" % cat_split_counter
                cat_split_counter += 1
                matching_keys = len([_ for _ in labels if _.startswith(prefix)])
                if matching_keys == 0:
                    break
                cat_feat_splits.append(matching_keys)
            continue
        count += 1
        cat_labels = []
        cat_features = []
        cont_labels = []
        cont_features = []
        id_labels = []
        id_features = []
        if len(labels) != len(row):
            print "CSV ERROR: label and row not same length (%s labels, %s lineitems), line %s" % (len(labels),len(row),count)
            continue
        values = zip(labels, row)
        for k,v in values:
            id_val = False
            cat_val = False
            cont_val = False
            if re.search(id_regex, k):
                k = k[k.index('#')+1:]
                id_val = True
            elif re.search(cat_regex, k):
                k = k[k.index('#')+1:]
                cat_val = True
            else:
                cont_val = True
            if id_val:
                try:
                    v = float(v)
                except Exception as e:
                    v = v
                id_labels.append(k)
                id_features.append(v)
            elif cat_val:
                try:
                    v = float(v)
                except Exception as e:
                    v = np.nan
                cat_labels.append(k)
                cat_features.append(v)
            elif cont_val:
                try:
                    v = float(v)
                except Exception as e:
                    v = np.nan
                cont_labels.append(k)
                cont_features.append(v)
        id_dict = dict(zip(id_labels, id_features))
        yield (id_dict, cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)
        count += 1





if __name__=="__main__":
    arg = {'player_id': 'biyombi01', 'team_id': 'CHA', 'target_game': '201404160CHA'}
    #arg = {'player_id': 'parketo01', 'team_id': 'SAS', 'target_game': '200910310SAS'}
    #arg = {'player_id': 'westda01', 'team_id': 'NOH', 'target_game': '201001290NOH'}
    #arg = {'player_id': 'curryst01', 'team_id': 'GSW', 'target_game': '201503180GSW'}
    #arg = {'player_id': 'warretj01', 'team_id': 'PHO', 'target_game': '201502250DEN'}
    #arg = {'player_id': 'tollian01', 'team_id': 'DET', 'target_game': '201503310DET'}
    invalids = []
    
    fe = NBAFeatureExtractor(arg['player_id'], arg['team_id'], target_game=arg['target_game'], invalids=invalids)
    cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = fe.runEncoderFeatures()
    
    print len(cont_features)
    print len(cont_labels)
    print len(cat_labels)
    print len(cat_features)


    for l,v in zip(cat_labels, cat_features):
        print "%s: %s" % (l,v)

    print '\n'
    for l,v in zip(cont_labels, cont_features):
        print "%s: %s" % (l,v)
    
    sys.exit(1)
    
    dumpCSV("/usr/local/fsai/analysis/data/nba_stats_2.csv", pid=None, limit=None, time=None, min_count=None, save=True)



