'''
Functions necessary to power API endpoints,
as well as feature vector objects for player-in-game
'''

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

def playerOffenseScatter(player_vectors):
    pass

def playerDefenseScatter(player_vectors):
    pass

def playerShotChartScatter(player_vectors):
    pass

def teamOffenseScatter(team_vectors):
    pass

def teamDefenseScatter(team_vectors):
    pass

def teamShotChartScatter(team_vectors):
    pass


class teamFeatureVector(object):

    '''
    team performance up until a certain day, given a stats window
    subsumed by playerFeatureVector
    '''


    TEAM_FRAMES = ['sqlTeamsMisc',
                   'sqlTeamsUsage',
                   'sqlTeamsFourFactors',
                   'sqlTeamsScoring',
                   'TeamStats',
                   'PlayerTrackTeam']


    def __init__(self, tid, gid, window=10):
        self.tid = int(tid)
        self.gid = str(gid)
        self.window = int(window)

        # find the game
        game_row = nba_games_collection.find_one({"_id": self.gid})
        if not game_row:
            raise Exception("Can't find game %s" % self.gid)
        self.game_row = game_row

        # get invalids, game info
        self.invalids = pd.read_json(self.game_row['InactivePlayers'])
        self.gamesummary = pd.read_json(self.game_row['GameSummary']).iloc[0]

        self.home_team = int(self.gamesummary['HOME_TEAM_ID'])
        self.away_team = int(self.gamesummary['VISITOR_TEAM_ID'])
        if self.tid not in set([self.home_team, self.away_team]):
            raise Exception("Team %s not in game %s" % (self.tid, self.gid))
        self.oid = self.home_team if self.away_team == self.tid else self.away_team

        # calculate date ranges
        self.end_date = self.game_row['date']- timedelta(days=1)
        start_year = self.end_date.year-1 if (self.end_date < datetime(day=1, month=10, year=self.end_date.year)) else self.end_date.year
        self.season_start = datetime(day=1, month=10, year=start_year)

        # find the team 
        team_row = nba_teams_collection.find_one({"team_id": self.tid, "season": start_year})
        team_row_opp = nba_teams_collection.find_one({"team_id": self.oid, "season": start_year})
        if not team_row:
            raise Exception("Can't find team %s" % self.tid)
        if not team_row_opp:
            raise Exception("Can't find opp team: %s" % self.oid)
        self.team_row = team_row
        self.team_row_opp = team_row_opp

        self.input = {}
        self.output = {}
        self.season_means = {}
        self.season_vars = {}

    def bootstrap(self, rows, samples, tid):
        # bootstrap sample
        size = len(rows)
        sampled_means = {k:[] for k in self.TEAM_FRAMES}
        sampled_vars = {k:[] for k in self.TEAM_FRAMES}
        for i in range(samples):
            bs_sample = np.random.choice(rows, size=size, replace=True)
            merged = self.mergeRows(bs_sample, tid)
            for k,v in merged.iteritems():
                sampled_means[k].append(dict(v.mean(skipna=True)))
                sampled_vars[k].append(dict(v.var(skipna=True)))
        # merge the samples
        for k,v in sampled_means.iteritems():
            sampled_means[k] = pd.DataFrame(v)
            print k
            print sampled_means[k]
        for k,v in sampled_vars.iteritems():
            sampled_vars[k] = pd.DataFrame(v)

        return (sampled_means, sampled_vars)

    def loadSeasonAverages(self, samples=20, recalculate=False):
        '''
        Get team averages over a longer window (bootstrapped)
        '''
        # try to find in database
        saved_query = {"date": self.end_date, "team_id": self.tid}
        saved = nba_season_averages_collection.find_one(saved_query)
        if saved and not recalculate:
            self.season_means = saved['means']
            self.season_vars = saved['vars']
            return

        # calculate it
        query = {"teams": self.tid, "date": {"$lte": self.end_date, "$gte": self.season_start}}
        season_rows = [self.parseGameRow(g) for g in nba_games_collection.find(query)]
        season_rows = [_ for _ in season_rows if _]
        if not season_rows:
            raise Exception("No season rows found for %s" % self.tid)

        # bootstrap sample
        sample_means, sample_vars = self.bootstrap(season_rows, samples, self.tid)

        # calculate averages
        self.season_means = {k: dict(v.mean(skipna=True)) for k,v in sample_means.iteritems()}
        self.season_vars = {k: dict(v.mean(skipna=True)) for k,v in sample_vars.iteritems()}

        # save/update
        data = {'means': self.season_means, 'vars': self.season_vars, 'team_id': self.tid, 'date': self.end_date}
        if saved:
            nba_conn.updateDocument(nba_season_averages_collection, saved['_id'], data, upsert=False)
        else:
            nba_conn.saveDocument(nba_season_averages_collection, data)

    def mergeRows(self, rows, tid):
        merged = {}
        for k in self.TEAM_FRAMES:
            all_rows = pd.concat([_[k] for _ in rows])
            merged[k] = all_rows[all_rows['TEAM_ID'] == tid]
        return merged

    def parseGameRow(self, g):
        parsed = {}
        success = True
        for k,v in g.iteritems():
            if k in self.TEAM_FRAMES:
                parsed[k] = pd.read_json(v)
                # check for empty
                if len(parsed[k].index) == 0:
                    success = False
                    break
        if success:
            return parsed
        return None

    def getWindowGames(self):
        parsed_window_games = []
        parsed_window_games_opp = []

        # get own stats
        query = {"teams": self.tid, "date": {"$lte": self.end_date, "$gte": self.season_start}}
        window_games = [self.parseGameRow(g) for g in nba_games_collection.find(query, sort=[("date",-1)], limit=self.window)]
        window_games = [_ for _ in window_games if _]
        
        # get opponent stats 
        query = {"teams": self.oid, "date": {"$lte": self.end_date, "$gte": self.season_start}}
        window_games_opp = [self.parseGameRow(g) for g in nba_games_collection.find(query, sort=[("date",-1)], limit=self.window)]
        window_games_opp = [_ for _ in window_games_opp if _]

        if not window_games:
            raise Exception("No previous games found for team %s, date %s" % (self.tid, self.end_date))
        if not window_games_opp:
            raise Exception("No previous games found for opp team %s, date %s" % (self.oid, self.end_date))

        return (window_games, window_games_opp)

    def loadInputs(self, samples=20, recalculate=False):
        # try to find in database
        saved_query = {"date": self.end_date, "team_id": self.tid}
        saved = nba_team_vectors_collection.find_one(saved_query)
        if saved and not recalculate:
            self.input = saved['input']
            return

        window_games, window_games_opp = self.getWindowGames()
        sample_means, sample_vars = self.bootstrap(window_games, samples, self.tid)
        sample_means_opp, sample_vars_opp = self.bootstrap(window_games_opp, samples, self.oid)
        
        # calculate averages 
        self.input['means'] = {k: dict(v.mean(skipna=True)) for k,v in sample_means.iteritems()}
        self.input['means_opp'] = {k: dict(v.mean(skipna=True)) for k,v in sample_means_opp.iteritems()}
        self.input['variances'] = {k: dict(v.mean(skipna=True)) for k,v in sample_vars.iteritems()}
        self.input['variances_opp'] = {k: dict(v.mean(skipna=True)) for k,v in sample_vars_opp.iteritems()}

        # parse game contextual features from game row and team row; location, home/away
        if self.home_team == self.tid:
            self.input['home/away'] = 'home'
            self.input['location'] = self.team_row['TEAM_CITY']
        else:
            self.input['home/away'] = 'away'
            self.input['location'] = self.team_row_opp['TEAM_CITY']

        # save/update
        data = {'input': self.input, 'team_id': self.tid, 'date': self.end_date}
        if saved:
            nba_conn.updateDocument(nba_team_vectors_collection, saved['_id'], data, upsert=False)
        else:
            nba_conn.saveDocument(nba_team_vectors_collection, data)


    def loadOutputs(self):
        # parse game row for outputs
        parsed = self.parseGameRow(self.game_row)
        if parsed:
            self.output.update(parsed)


class playerFeatureVector(object):

    PLAYER_FRAMES = ['PlayerTrack',
                     'PlayerStats',
                     'sqlPlayersUsage',
                     'OtherStats',
                     'sqlPlayersScoring',
                     'sqlPlayersFourFactors',
                     'sqlPlayersMisc']


    def __init__(self, pid, gid, own_team_vector, opp_team_vector):
        self.pid = int(pid)
        self.gid = str(gid)

        self.own_team_vector = own_team_vector
        self.opp_team_vector = opp_team_vector

        # make sure team vectors match
        assert (self.gid == self.own_team_vector.gid)
        assert (self.gid == self.opp_team_vector.gid)

        self.input={}
        self.output={}
        self.season_trends={}

        self.loadInputs()
        self.loadOutputs()
        self.loadSeasonTrends()


    def bayesianBootstrap(self, rows):
        pass

    def parseGameRow(self, g):
        parsed = {}
        success = True
        for k,v in g.iteritems():
            if k in self.PLAYER_FRAMES:
                parsed[k] = pd.read_json(v)
                # check for empty
                if len(parsed[k].index) == 0:
                    success = False
                    break
        if success:
            return parsed
        return None

    def mergeRows(self, rows, pid):
        merged = {}
        for k in self.PLAYER_FRAMES:
            all_rows = pd.concat([_[k] for _ in rows])
            merged[k] = all_rows[all_rows['PLAYER_ID'] == pid]
        return merged

    def getWindowGames(self):
        '''
        get opponent team vectors for the windowed games as well (for strength of schedule)
        '''
        # get own stats
        query = {"teams": self.own_team_vector.tid, "date": {"$lte": self.own_team_vector.end_date, "$gte": self.own_team_vector.season_start}}
        window_games = [self.parseGameRow(g) for g in nba_games_collection.find(query, sort=[("date",-1)], limit=self.own_team_vector.window)]
        window_games = [_ for _ in window_games if _]
        return window_games

    def imputeRow(self, row):
        '''
        Make necessary conversions/calcutions:
        - Fill in MIN=0.0
        - Calculate OffRtg, DefRtg, Poss
        '''
        for k,v in row.iteritems():
            # fill in zero minutes
            if 'MIN' in v:
                v['MIN'] = v['MIN'].fillna(value='0:0').str.split(':').apply(lambda x: float(x[0]) + float(x[1])/60.0)
            # calculate possessions
            # calculate offensive rating
            # calculate defensive rating

            row[k] = v
        return row

    def loadSeasonTrends(self, recalculate=False):
        '''
        find similar player/game contexts in recent memory (across some key high-level team and individual stats)
        calculate trends for current game context for a longer window (home/away/daysrest/similar opponents (maybe across similar players as well))
        Stratify, then sample
        '''
        pass

    def strengthOfSchedule(self, opp_args):
        '''
        Find the team vectors corresponding to team_vector_args
        Get season stats from each vector
        Calculate performance above average player against schedule
        '''
        vectors = []
        for gid, tid in opp_args:
            pass




    def loadInput(self, recalculate=False):
        # parse own stats
        window_games = self.getWindowGames()
        most_recent = window_games[0]['date']
        opp_args = []
        for _ in window_games:
            opp_team = _['teams'][0] if (_['teams'][0] != self.own_team_vector.tid) else _['teams'][1]
            opp_args.append(_['_id'],int(opp_team))
        merged = self.mergeRows(window_games, self.pid)
        merged = self.imputeRow(merged)

        # calculate averages
        self.input['means'] = {k: v.mean(skipna=True) for k,v in merged.iteritems()}
        self.input['means_opp'] = {k: v.mean(skipna=True) for k,v in merged_opp.iteritems()}
        self.input['variances'] = {k: v.var(skipna=True) for k,v in merged.iteritems()}
        self.input['variances_opp'] = {k: v.var(skipna=True) for k,v in merged_opp.iteritems()}

        # parse contextual stats (minutes played, days rest, etc)
        self.input['days_rest'] = (self.own_team_vector.game_row['date'] - most_recent).days

        # pull up relevant team vector features
        self.input['home/away'] = self.own_team_vector.input['home/away']
        self.input['location'] = self.own_team_vector.input['location']

        # pull up season stats from previous team vectors for strength of schedule
        self.input['strengthOfSchedule'] = self.strengthOfSchedule(opp_args)


    def loadOutput(self):
        '''
        parse game row for player performance (for historical games only)
        '''
        # parse game row for outputs
        parsed = self.parseGameRow(self.own_team_vector.game_row)
        if parsed:
            self.output.update(parsed)




if __name__=='__main__':
    gid = "0021400242"
    tid = "1610612741"
    v = teamFeatureVector(tid, gid)
    v.loadInputs(recalculate=True)
    #v.loadOutputs()
    #v.loadSeasonAverages(recalculate=True)

    print v.input

