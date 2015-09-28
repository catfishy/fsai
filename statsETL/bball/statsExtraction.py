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

class featureVector():

    def __init__(self):
        pass

    def getPlayerRow(self, pid):
        # get position from player row
        query = {"_id": pid}
        player_row = nba_conn.findOne(nba_players_collection, query=query)
        if not player_row:
            raise Exception("Can't find player %s" % pid) 

    def getPlayerPosition(self, pid):
        player_row = getPlayerRow(pid)
        fullpos = player_row['POSITION'].lower()
        if 'forward' in fullpos:
            pos = 'F'
        if 'center' in fullpos:
            pos = 'C'
        if 'guard' in fullpos:
            pos = 'G'
        else:
            raise Exception("Couldn't find player position in db row: %s, %s" % (pid, fullpos))


    def serialize(self, data):
        for k,v in data.iteritems():
            if type(v) == pd.DataFrame:
                v = v.to_json()
            data[k] = v
        return data

    def deserialize(self, data):
        for k,v in data.iteritems():
            try:
                v = pd.read_json(v)
            except Exception as e:
                pass
            data[k] = v
        return data

    def bootstrap(self, data, num_samples, statistic):
        """
        Returns the results from num_samples bootstrap samples
        """
        # Generate the indices for the required number of permutations/(resamplings with replacement) required
        idx = np.random.randint(0, len(data.index), (num_samples, len(data)))
        stats = [statistic(data.ix(idx_row)) for idx_row in idx]
        # Generate the multiple resampled data set from the original one
        samples = pd.concat(stats,axis=1).transpose()
        description = samples.describe()
        return description

    def bayesianBootstrap(self, data, num_samples, statistic, samplesize):
        def Dirichlet_sample(m,n):
            """Returns a matrix of values drawn from a Dirichlet distribution with parameters = 1.
            'm' rows of values, with 'n' Dirichlet draws in each one."""
            # Draw from Gamma distribution
            Dirichlet_params = np.ones(m*n) # Set Dirichlet distribution parameters
            # https://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution
            Dirichlet_weights = np.asarray([random.gammavariate(a,1) for a in Dirichlet_params])
            Dirichlet_weights = Dirichlet_weights.reshape(m,n) # Fold them (row by row) into a matrix
            row_sums = Dirichlet_weights.sum(axis=1)
            Dirichlet_weights = Dirichlet_weights / row_sums[:, np.newaxis] # Reweight each row to be normalised to 1
            return Dirichlet_weights

        Dirich_wgts_matrix = Dirichlet_sample(num_samples, data.shape[0]) #Generate sample of Dirichlet weights
        
        # If statistic can be directly computed using the weights (such as the mean), do this since it will be faster.
        # TODO: CHECK IF THIS WORKS
        if statistic in set([np.nanmean, np.mean, np.average]):
            results = pd.concat([statistic(data, weights=Dirich_wgts_matrix[i]) for i in xrange(num_samples)],axis=1).transpose()
            return results.describe()
        else: # Otherwise resort to sampling according to the Dirichlet weights and computing the statistic
            results = []
            for i in xrange(num_samples): #Sample from data according to Dirichlet weights
                idx_row = np.random.choice(range(len(data.idx)), samplesize, replace=True, p = Dirich_wgts_matrix[i])
                weighted_sample = data.ix(idx_row)
                results.append(statistic(weighted_sample)) #Compute the statistic for each sample
            results = pd.concat(results, axis=1).transpose()
            return results.describe()


    


class teamFeatureVector(featureVector):

    '''
    team performance up until a certain day, given a stats window
    subsumed by playerFeatureVector
    '''


    TEAM_FRAMES = ['sqlTeamsMisc',
                   'sqlTeamsUsage',
                   'sqlTeamsFourFactors',
                   'sqlTeamsScoring',
                   'TeamStats',
                   'PlayerTrackTeam',
                   'OtherStats',]


    def __init__(self, tid, gid, window=10, recalculate=False):
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

        self.output = None
        self.loadOutput()

        self.input = None
        self.loadInput(recalculate=recalculate)

        self.season_stats = None
        self.loadSeasonAverages(recalculate=recalculate)

    def loadSeasonAverages(self, samples=20, recalculate=False):
        '''
        Get team averages over a longer window (bootstrapped)
        '''
        # try to find in database
        saved_query = {"date": self.end_date, "team_id": self.tid}
        saved = nba_season_averages_collection.find_one(saved_query)
        if saved and not recalculate:
            self.season_stats = self.deserialize(saved['season_stats'])
            return

        self.season_stats = {}

        # calculate it
        query = {"teams": self.tid, "date": {"$lte": self.end_date, "$gte": self.season_start}}
        season_rows = [self.parseGameRow(g) for g in nba_games_collection.find(query)]
        season_rows = [_ for _ in season_rows if _]
        if not season_rows:
            raise Exception("No season rows found for %s" % self.tid)

        # bootstrap sample
        season_means = self.bayesianBootstrap(season_rows, samples, np.nanmean, samplesize=20)
        season_vars = self.bayesianBootstrap(season_rows, samples, np.nanvar, samplesize=20)
        self.season_stats['means'] = season_means
        self.season_stats['vars'] = season_vars

        # save/update
        data = {'season_stats': self.serialize(self.season_stats), 'team_id': self.tid, 'date': self.end_date}

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

    def imputeRows(self, row):
        # merge frames
        merged = None
        for k,v in row.iteritems():
            if k in self.TEAM_FRAMES:
                if merged is None:
                    merged = v
                    continue
                else:
                    cols_to_use = (v.columns-merged.columns).tolist()
                    cols_to_use.append('GAME_ID')
                    merged = pd.merge(merged, v[cols_to_use], on='GAME_ID')

        if 'MIN' in merged:
            merged['MIN'] = merged['MIN'].fillna(value='0:0').str.split(':').apply(lambda x: float(x[0]) + float(x[1])/60.0)

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

    def loadInput(self, samples=20, recalculate=False):
        # try to find in database
        saved_query = {"date": self.end_date, "team_id": self.tid, "window": self.window}
        saved = nba_team_vectors_collection.find_one(saved_query)
        if saved and not recalculate:
            self.input = self.deserialize(saved['input'])
            return

        self.input = {}

        window_games, window_games_opp = self.getWindowGames()
        window_games = self.imputeRows(self.mergeRows(window_games))
        window_games_opp = self.imputeRows(self.mergeRows(window_games_opp))

        sample_means = self.bootstrap(window_games, samples, np.nanmean)
        sample_means_opp = self.bootstrap(window_games, samples, np.nanvar)
        sample_vars = self.bootstrap(window_games_opp, samples, np.nanmean)
        sample_vars_opp = self.bootstrap(window_games_opp, samples, np.nanvar)

        # calculate averages 
        self.input['means'] = sample_means
        self.input['means_opp'] = sample_means_opp
        self.input['variances'] = sample_vars
        self.input['variances_opp'] = sample_vars_opp

        # parse game contextual features from game row and team row; location, home/road
        if self.home_team == self.tid:
            self.input['home/road'] = 'home'
            self.input['location'] = self.team_row['TEAM_CITY']
        else:
            self.input['home/road'] = 'road'
            self.input['location'] = self.team_row_opp['TEAM_CITY']

        # save/update
        data = {'input': self.serialize(self.input), 'team_id': self.tid, 'date': self.end_date, 'window': self.window}
        if saved:
            nba_conn.updateDocument(nba_team_vectors_collection, saved['_id'], data, upsert=False)
        else:
            nba_conn.saveDocument(nba_team_vectors_collection, data)


    def loadOutput(self):
        # parse game row for outputs
        parsed = self.parseGameRow(self.game_row)
        if not parsed:
            print "Game Row could not be parsed: possible bad game or future game"
            self.output = {}
        else:
            merged = self.mergeRows([parsed], self.tid)
            data = self.imputeRows(merged)
            self.output = data


class playerFeatureVector(featureVector):

    PLAYER_FRAMES = ['PlayerTrack',
                     'PlayerStats',
                     'sqlPlayersUsage',
                     'sqlPlayersScoring',
                     'sqlPlayersFourFactors',
                     'sqlPlayersMisc']

    TREND_KEYS = []

    def __init__(self, pid, gid, own_team_vector, opp_team_vector):
        self.pid = int(pid)
        self.gid = str(gid)

        self.own_team_vector = own_team_vector
        self.opp_team_vector = opp_team_vector

        # make sure team vectors match
        assert (self.gid == self.own_team_vector.gid)
        assert (self.gid == self.opp_team_vector.gid)
        assert (self.own_team_vector.window == self.opp_team_vector.window)
        self.window = self.own_team_vector.window
        self.long_window = self.window*2

        self.output=None
        self.loadOutput()

        self.input=None
        self.loadInput()

    def getGamePosition(self):
        '''
        Get the player's position in the specified game
        '''
        if self.output is None:
            # TODO: specify contingency for future games
            raise Exception("Can't load future positions yet!")
        player_row = self.output[self.output['PLAYER_ID' == self.pid]]
        player_pos = str(player_row['START_POSITION']).strip()
        if player_pos == '':
            playerpos = self.getPlayerPosition(pid)
        return player_pos


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
                # check for advanced stats
                if k == 'PlayerStats':
                    if 'OFF_RATING' not in parsed[k]:
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
        window_games = [self.parseGameRow(g) for g in nba_games_collection.find(query, sort=[("date",-1)], limit=self.window)]
        window_games = [_ for _ in window_games if _]
        return window_games

    def imputeRows(self, row):
        '''
        Make necessary conversions/calcutions:
        - Fill in MIN=0.0
        - Calculate OffRtg, DefRtg, Poss
        '''
        # merge frames
        merged = None
        for k,v in row.iteritems():
            if k in self.PLAYER_FRAMES:
                if merged is None:
                    merged = v
                    continue
                else:
                    cols_to_use = (v.columns-merged.columns).tolist()
                    cols_to_use.append('GAME_ID')
                    merged = pd.merge(merged, v[cols_to_use], on='GAME_ID')

        if 'MIN' in merged:
            merged['MIN'] = merged['MIN'].fillna(value='0:0').str.split(':').apply(lambda x: float(x[0]) + float(x[1])/60.0)

        return merged

    def loadSeasonSplits(self, samples, recalculate=False):
        saved_query = {"date": self.own_team_vector.end_date, "player_id": self.pid}
        saved = nba_split_vectors_collection.find_one(saved_query)
        if saved and not recalculate:
            return self.deserialize(saved['input'])
        
        # calculate splits
        splits = {'home/road': [],
                  'days/rest': [],
                  'start/bench': [],
                  'against_team': []}




    def loadSeasonTrends(self, player_data, against_data, samples):
        '''
        calculate how the player has performed against average performance against opponent
        '''
        trends = []
        for i, row in player_data.iterrows():
            gid = row['GAME_ID']
            tid = row['TEAM_ID']
            against_row = against_data[(gid, tid)]
            mean = against_row['mean'].ix['mean']
            var = against_row['var'].ix['mean']
            trend = (row-mean)/np.sqrt(std)
            trends.append(trend.ix[self.TREND_KEYS])
        alltrends = pd.concat(trends, axis=1).transpose()
        mean_distr = self.bayesianBootstrap(alltrends, samples, np.nanmean, samplesize=20)
        var_distr = self.bayesianBootstrap(alltrends, samples, np.nanvar, samplesize=20)
        return (mean_distr, var_distr)


    def performanceAgainstTeam(self, opp_args, samples, recalculate=False):
        '''
        Get from DB or recalculate average performance against opposition (bayesian bootstrap)
        To define a neighborhood: sample position players who played more than 10 min in game
        '''
        data = {}
        # get stats ubset (normalized) from players who have played similar position against each team
        for gid, tid in opp_args:
            # look in db
            saved_query = {"game_id": pid, "team_id": tid}
            saved = nba_against_vectors_collection.find_one(saved_query)
            if saved and not recalculate:
                data[(gid, tid)] = self.deserialize(saved['input'])
                continue

            # get game row
            vector = teamFeatureVector(tid, gid, window=self.long_window)
            # get position of players
            stats = vector.game_row['PlayerStats']
            opp_player_rows = stats[stats['TEAM_ID'] == vector.oid]
            by_position = {'G': [],
                           'F': [],
                           'C': []}
            for i, row in opp_player_rows.iterrows():
                pid = int(row['PLAYER_ID'])
                pos = str(row['START_POSITION']).strip()
                if pos == '':
                    pos = self.getPlayerPosition(pid)
                # add stats
                by_position[pos].append(row)
            # join into dataframes
            for k,v in by_position.iteritems():
                allrows = pd.concat(v)
                mean_distr = self.bayesianBootstrap(allrows, samples, np.nanmean, samplesize=20)
                var_distr = self.bayesianBootstrap(allrows, samples, np.nanvar, samplesize=20)
                by_position[k] = {"mean": mean, "var": var}
            data[(gid, tid)] = by_position

            # save
            to_save = {"input": self.serialize(by_position), "game_id": gid, "team_id": tid}
            if saved:
                nba_conn.updateDocument(nba_against_vectors_collection, saved['_id'], to_save, upsert=False)
            else:
                nba_conn.saveDocument(nba_against_vectors_collection, to_save)

        return data


    def loadInput(self, samples=20, recalculate=False):
        # try to find in database
        saved_query = {"date": self.end_date, "player_id": self.pid, "window": self.window}
        saved = nba_player_vectors_collection.find_one(saved_query)
        if saved and not recalculate:
            self.input = self.deserialize(saved['input'])
            return

        self.input={}

        # parse own stats
        window_games = self.getWindowGames()
        most_recent = window_games[0]['date']
        opp_args = [(self.gid, self.own_team_vector.oid)]
        for _ in window_games:
            opp_team = _['teams'][0] if (_['teams'][0] != self.own_team_vector.tid) else _['teams'][1]
            opp_args.append(_['_id'],int(opp_team))
        merged = self.mergeRows(window_games, self.pid)
        data = self.imputeRow(merged)

        # calculate averages
        sample_means = self.bootstrap(data, samples, np.nanmean)
        sample_vars = self.bootstrap(data, samples, np.nanvar)
        self.input['means'] = sample_means
        self.input['variances'] = sample_vars

        # parse contextual stats (minutes played, days rest, etc)
        self.input['days_rest'] = (self.own_team_vector.game_row['date'] - most_recent).days

        # pull up relevant team vector features
        self.input['home/road'] = self.own_team_vector.input['home/road']
        self.input['location'] = self.own_team_vector.input['location']

        self.input['position'] = self.getGamePosition()

        # calculate splits
        split_means, split_vars = self.loadSeasonSplits(samples, recalculate=recalculate)
        self.input['split_means'] = split_means
        self.input['split_variances'] = split_vars

        # calculate opposition strength
        against_data = self.performanceAgainstTeam(opp_args, samples, recalculate=recalculate)
        against_row = against_data[(self.gid, self.own_team_vector.oid)][self.input['position']]
        self.input['against_mean'] = against_row['mean']
        self.input['against_var'] = against_row['var']

        # calculate trends
        trend_means, trend_vars = self.loadSeasonTrends(data, against_data, samples)
        self.input['trend_means'] = trend_means
        self.input['trend_variances'] = trend_vars

        # save/update
        data = {'input': self.serialize(self.input), 'player_id': self.pid, 'date': self.end_date, 'window': self.window}
        if saved:
            nba_conn.updateDocument(nba_player_vectors_collection, saved['_id'], data, upsert=False)
        else:
            nba_conn.saveDocument(nba_player_vectors_collection, data)

    def loadOutput(self):
        '''
        parse game row for player performance (for historical games only)
        '''
        # parse game row for outputs
        parsed = self.parseGameRow(self.own_team_vector.game_row)
        if not parsed:
            print "Game Row could not be parsed: possible bad game or future game"
            self.output = {}
        else:
            merged = self.mergeRows([parsed], self.pid)
            data = self.imputeRows(merged)
            self.output = data




if __name__=='__main__':
    gid = '0021400585'
    tid = "1610612752"
    v = teamFeatureVector(tid, gid)
    v.loadInput(recalculate=True)
    v.loadOutput()
    #v.loadSeasonAverages(recalculate=True)

    print v.input

