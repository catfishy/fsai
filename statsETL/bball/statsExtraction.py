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
import random
import copy

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

from statsETL.db.mongolib import *
from statsETL.db.cachelib import *


WINDOW = 10


def getPlayerVector(pid, tid, gid):
    key = (pid, tid, gid)

    if not recalculate:
        result = findKey(key)
        if result:
            return result

    vector = playerFeatureVector(
        pid, tid, gid, window=WINDOW, recalculate=recalculate)
    result = vector.input

    # Place in cache
    if recalculate:
        updated = updateCache(key, result)

    return result


def getTeamVector(tid, gid, recalculate=False):
    key = (tid, gid)

    if not recalculate:
        result = findKey(key)
        if result:
            return result

    vector = teamFeatureVector(
        tid, gid, window=WINDOW, recalculate=recalculate)
    result = vector.input

    # Place in cache
    if recalculate:
        updated = updateCache(key, result)

    return result


def df_mean(df, weights=None):
    df = df.convert_objects(convert_numeric=True)
    df = df._get_numeric_data()
    if weights is not None:
        weights = pd.Series(weights)
        weights = weights / weights.sum()
        for c in df.columns:
            try:
                df[c] = df[c] * weights
            except:
                print "Can't weigh column %s" % c
        return df.sum(skipna=True)
    else:
        return df.mean(skipna=True)


def df_var(df):
    df = df.convert_objects(convert_numeric=True)
    df = df._get_numeric_data()
    return df.var(skipna=True)


class featureVector():

    def __init__(self):
        pass

    def aggregateShotChart(self, query):
        '''
        Parse into percentages from shot zones
        '''
        shot_chart_keys = [(u'Mid-Range', u'Center(C)', u'16-24 ft.'),
                           (u'Mid-Range', u'Right Side(R)', u'16-24 ft.'),
                           (u'Left Corner 3', u'Left Side(L)', u'24+ ft.'),
                           (u'Mid-Range', u'Right Side(R)', u'8-16 ft.'),
                           (u'In The Paint (Non-RA)', u'Center(C)', u'8-16 ft.'),
                           (u'Right Corner 3', u'Right Side(R)', u'24+ ft.'),
                           (u'Restricted Area', u'Center(C)', u'Less Than 8 ft.'),
                           (u'Mid-Range', u'Right Side Center(RC)', u'16-24 ft.'),
                           (u'Mid-Range', u'Left Side Center(LC)', u'16-24 ft.'),
                           (u'In The Paint (Non-RA)',
                            u'Center(C)', u'Less Than 8 ft.'),
                           (u'In The Paint (Non-RA)',
                            u'Right Side(R)', u'8-16 ft.'),
                           (u'Backcourt', u'Back Court(BC)', u'Back Court Shot'),
                           (u'In The Paint (Non-RA)', u'Left Side(L)', u'8-16 ft.'),
                           (u'Above the Break 3', u'Center(C)', u'24+ ft.'),
                           (u'Mid-Range', u'Center(C)', u'8-16 ft.'),
                           (u'Mid-Range', u'Left Side(L)', u'8-16 ft.'),
                           (u'Mid-Range', u'Left Side(L)', u'16-24 ft.'),
                           (u'Above the Break 3', u'Right Side Center(RC)', u'24+ ft.'),
                           (u'Above the Break 3', u'Left Side Center(LC)', u'24+ ft.'),
                           (u'Above the Break 3', u'Back Court(BC)', u'Back Court Shot')]
        print query
        shots=shot_chart_collection.find(query)
        by_zones_made={_:0 for _ in shot_chart_keys}
        by_zones_attempted={_:0 for _ in shot_chart_keys}
        by_zones_percent = {_:np.nan for _ in shot_chart_keys}
        for sc in shots:
            zone_basic=sc["SHOT_ZONE_BASIC"]
            zone_area=sc['SHOT_ZONE_AREA']
            zone_range=sc['SHOT_ZONE_RANGE']
            zone=(zone_basic, zone_area, zone_range)
            made=True if sc["EVENT_TYPE"] == "Made Shot" else False
            if made:
                by_zones_made[zone] += 1
            by_zones_attempted[zone] += 1
        for k,v in by_zones_attempted.iteritems():
            if v > 0:
                by_zones_percent[k] = float(by_zones_made[k])/float(v)

        # collapse
        data = {}
        for k in shot_chart_keys:
            k_str = ' '.join(list(k)).replace('.','').replace(' ','_').replace('-','_')
            data["%s_attempted" % k_str] = by_zones_attempted[k]
            data["%s_percent" % k_str] = by_zones_percent[k]
        return data

    def getPlayerRow(self, pid):
        # get position from player row
        query={"_id": int(pid)}
        player_row=nba_conn.findOne(nba_players_collection, query=query)
        if not player_row:
            raise Exception("Can't find player %s" % pid)
        return player_row

    def getPlayerPosition(self, pid):
        player_row=self.getPlayerRow(pid)
        fullpos=str(player_row['POSITION']).lower().strip()
        br_pos=[_.strip() for _ in player_row.get('BR_POSITION', [])]
        pos=[]
        if 'forward' in fullpos:
            pos.append('F')
        if 'center' in fullpos:
            pos.append('C')
        if 'guard' in fullpos:
            pos.append('G')
        if len(br_pos) > 0:
            if 'PG' in br_pos or 'SG' in br_pos:
                pos.append('G')
            if 'SF' in br_pos or 'PF' in br_pos:
                pos.append('F')
            if 'C' in br_pos:
                pos.append('C')
        pos=list(set(pos))
        if len(pos) == 0:
            raise Exception(
                "Couldn't find player position in db row: %s, %s, %s" % (pid, fullpos, br_pos))
        return pos

    def getGameRow(self, gid):
        '''
        Possibly a future game
        Validate that necessary fields exist
        '''
        # find the game
        game_row=nba_games_collection.find_one({"_id": gid})
        if not game_row:
            raise Exception("Can't find game %s" % gid)
        for k, v in game_row.iteritems():
            try:
                v=pd.read_json(v)
            except:
                pass
            game_row[k]=v
        return game_row

    def serialize(self, data):
        for k, v in data.iteritems():
            if isinstance(v, pd.DataFrame):
                v=v.to_json()
            if isinstance(v, dict):
                v=self.serialize(v)
            data[k]=v
        return data

    def deserialize(self, data):
        for k, v in data.iteritems():
            if isinstance(v, dict):
                v=self.deserialize(v)
            else:
                try:
                    v=pd.read_json(v)
                except Exception as e:
                    pass
            data[k]=v
        return data

    def exponentiallyWeightedMean(self, data):
        df=data.copy()
        df=df.convert_objects(convert_numeric=True)
        # set game date as index
        df=df.set_index(['GAME_DATE'], drop=False,
                          append=False, inplace=False, verify_integrity=False)
        df=df.sort_index(axis=0)
        df=df._get_numeric_data()
        result=pd.ewma(df, span=len(df.index), adjust=True,
                         min_periods=len(df.index), ignore_na=True)
        result_row=result.tail(1)
        result_row=result_row.reset_index(drop=True)
        return result_row

    def exponentiallyWeightedVar(self, data):
        df=data.copy()
        df=df.convert_objects(convert_numeric=True)
        # set game date as index
        df=df.set_index(['GAME_DATE'], drop=False,
                          append=False, inplace=False, verify_integrity=False)
        df=df.sort_index(axis=0)
        df=df._get_numeric_data()
        result=result=pd.ewmvar(df, span=len(
            df.index), adjust=True, min_periods=len(df.index), ignore_na=True)
        result_row=result.tail(1)
        result_row=result_row.reset_index(drop=True)
        return result_row

    def bootstrap(self, data, num_samples, statistic):
        """
        Returns the results from num_samples bootstrap samples
        """
        # Generate the indices for the required number of
        # permutations/(resamplings with replacement) required
        data=data.copy()
        print "Bootstrap: %s, %s" % (statistic, data.shape)
        idx=np.random.randint(0, len(data.index), (num_samples, len(data)))
        stats=[statistic(data.iloc[idx_row]) for idx_row in idx]
        # Generate the multiple resampled data set from the original one
        samples=pd.concat(stats, axis=1).transpose()
        description=samples.describe()
        return description

    def bayesianBootstrap(self, data, num_samples, statistic, samplesize):
        def Dirichlet_sample(m, n):
            """Returns a matrix of values drawn from a Dirichlet distribution with parameters = 1.
            'm' rows of values, with 'n' Dirichlet draws in each one."""
            # Draw from Gamma distribution
            # Set Dirichlet distribution parameters
            Dirichlet_params=np.ones(m * n)
            # https://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution
            Dirichlet_weights=np.asarray(
                [random.gammavariate(a, 1) for a in Dirichlet_params])
            Dirichlet_weights=Dirichlet_weights.reshape(
                m, n)  # Fold them (row by row) into a matrix
            row_sums=Dirichlet_weights.sum(axis=1)
            # Reweight each row to be normalised to 1
            Dirichlet_weights=Dirichlet_weights / row_sums[:, np.newaxis]
            return Dirichlet_weights

        data=data.copy()
        print "Bayesian Bootstrap: %s, %s" % (statistic, data.shape)
        # Generate sample of Dirichlet weights
        Dirich_wgts_matrix=Dirichlet_sample(num_samples, data.shape[0])

        # If statistic can be directly computed using the weights (such as the mean), do this since it will be faster.
        # TODO: CHECK IF THIS WORKS
        results=[]
        if statistic in set([df_mean]):
            for i in xrange(num_samples):
                new_sample=statistic(data, weights=Dirich_wgts_matrix[i])
                results.append(new_sample)
        else:  # Otherwise resort to sampling according to the Dirichlet weights and computing the statistic
            for i in xrange(num_samples):  # Sample from data according to Dirichlet weights
                idx_row=np.random.choice(
                    range(len(data.index)), samplesize, replace=True, p=Dirich_wgts_matrix[i])
                new_sample=statistic(data.iloc[idx_row])
                results.append(new_sample)
        results=pd.concat(results, axis=1).transpose()
        return results.describe()


class teamFeatureVector(featureVector):

    '''
    team performance up until a certain day, given a stats window
    subsumed by playerFeatureVector
    '''

    TEAM_FRAMES=['sqlTeamsMisc',
                   'sqlTeamsUsage',
                   'sqlTeamsFourFactors',
                   'sqlTeamsScoring',
                   'TeamStats',
                   'PlayerTrackTeam',
                   'OtherStats', ]

    def __init__(self, tid, gid, window=10, recalculate=False):
        self.tid=int(tid)
        self.gid=str(gid)
        self.window=int(window)

        # find the game
        self.game_row=self.getGameRow(self.gid)

        # get invalids, game info
        self.invalids=self.game_row['InactivePlayers']
        self.gamesummary=self.game_row['GameSummary'].iloc[0]

        self.home_team=int(self.gamesummary['HOME_TEAM_ID'])
        self.away_team=int(self.gamesummary['VISITOR_TEAM_ID'])
        if self.tid not in set([self.home_team, self.away_team]):
            raise Exception("Team %s not in game %s" % (self.tid, self.gid))
        self.oid=self.home_team if self.away_team == self.tid else self.away_team

        # calculate date ranges
        self.end_date=min(datetime.now(), self.game_row[
                            'date'] - timedelta(days=1))
        start_year=self.end_date.year - 1 if (self.end_date < datetime(
            day=1, month=10, year=self.end_date.year)) else self.end_date.year
        self.season_start=datetime(day=1, month=10, year=start_year)

        # find the team
        team_row=nba_teams_collection.find_one(
            {"team_id": self.tid, "season": start_year})
        team_row_opp=nba_teams_collection.find_one(
            {"team_id": self.oid, "season": start_year})
        if not team_row:
            raise Exception("Can't find team %s" % self.tid)
        if not team_row_opp:
            raise Exception("Can't find opp team: %s" % self.oid)
        self.team_row=team_row
        self.team_row_opp=team_row_opp

        self.output=None
        self.loadOutput()

        self.input=None
        self.loadInput(recalculate=recalculate)

        self.season_stats=None
        self.loadSeasonAverages(recalculate=recalculate)

    def loadSeasonAverages(self, samples=30, recalculate=False):
        '''
        Get team averages over a longer window (bootstrapped)
        '''
        # try to find in database
        saved_query={"date": self.end_date, "team_id": self.tid}
        saved=nba_season_averages_collection.find_one(saved_query)
        if saved and not recalculate:
            self.season_stats=self.deserialize(saved['season_stats'])
            return

        self.season_stats={}

        # calculate it
        query={"teams": self.tid, "date": {
            "$lte": self.end_date, "$gte": self.season_start}}
        season_rows=[self.parseGameRow(g)
                       for g in nba_games_collection.find(query)]
        season_rows=[_ for _ in season_rows if _]
        if not season_rows:
            raise Exception("No season rows found for %s" % self.tid)

        # bootstrap sample
        season_games=self.imputeRows(self.mergeRows(season_rows, self.tid))
        season_means=self.bayesianBootstrap(
            season_games, samples, df_mean, samplesize=samples)
        season_vars=self.bayesianBootstrap(
            season_games, samples, df_var, samplesize=samples)
        self.season_stats['means']=season_means
        self.season_stats['vars']=season_vars

        # save/update
        data={'season_stats': self.serialize(copy.deepcopy(
            self.season_stats)), 'team_id': self.tid, 'date': self.end_date}

        if saved:
            nba_conn.updateDocument(nba_season_averages_collection, saved[
                                    '_id'], data, upsert=False)
        else:
            nba_conn.saveDocument(nba_season_averages_collection, data)

    def mergeRows(self, rows, tid):
        merged={}
        for k in self.TEAM_FRAMES:
            all_rows=pd.concat([_[k] for _ in rows])
            merged[k]=all_rows[all_rows['TEAM_ID'] == tid]
        return merged

    def imputeRows(self, row, join_shot_charts=True):
        col_blacklist=['LEAGUE_ID',
                       'MIN']
        # merge frames
        merged=None
        for k, v in row.iteritems():
            if k in self.TEAM_FRAMES:
                if merged is None:
                    merged=v
                    continue
                else:
                    cols_to_use=(v.columns - merged.columns).tolist()
                    cols_to_use.append('GAME_ID')
                    merged=pd.merge(merged, v[cols_to_use], on='GAME_ID')

        # join shot charts
        shot_chart_rows = []
        if join_shot_charts:
            for i, row in merged.iterrows():
                query = {"game_id": str(row['GAME_ID']).zfill(10), "TEAM_ID": row['TEAM_ID']}
                sc_for_game = self.aggregateShotChart(query)
                sc_for_game['GAME_ID'] = row['GAME_ID']
                shot_chart_rows.append(sc_for_game)
            shot_chart_df = pd.DataFrame(shot_chart_rows)
            cols_to_use = (shot_chart_df.columns-merged.columns).tolist()
            cols_to_use.append('GAME_ID')
            merged = pd.merge(merged, shot_chart_df[cols_to_use], on='GAME_ID')

        if 'MIN' in merged:
            merged['MIN']=merged['MIN'].fillna(value='0:0').str.split(
                ':').apply(lambda x: float(x[0]) + float(x[1]) / 60.0)

        # remove blacklist
        for c in col_blacklist:
            if c in merged:
                merged.drop(c, axis=1, inplace=True)

        return merged

    def parseGameRow(self, g):
        parsed={}
        success=True
        gid=str(g['_id'])
        date=g['date']
        for k, df in g.iteritems():
            if k in self.TEAM_FRAMES:
                if isinstance(df, str) or isinstance(df, unicode):
                    df=pd.read_json(df)
                # check for empty
                if len(df.index) == 0:
                    success=False
                    break

                # sometimes tracking is empty, if DIST is 0, then set other
                # zero tracking stats to nan
                if k == 'PlayerTrackTeam':
                    zero_columns=[c for c in df.columns if df[c].sum() == 0]
                    if 'DIST' in zero_columns:
                        for c in zero_columns:
                            df[c]=np.nan

                # add game id if needed
                if 'GAME_ID' not in df:
                    df['GAME_ID']=int(gid)
                df['GAME_DATE']=g['date']

                parsed[k]=df


        if success:
            return parsed
        return None

    def getWindowGames(self):
        parsed_window_games=[]
        parsed_window_games_opp=[]

        # get own stats
        query={"teams": self.tid, "date": {
            "$lte": self.end_date, "$gte": self.season_start}}
        window_games=[self.parseGameRow(g) for g in nba_games_collection.find(
            query, sort=[("date", -1)], limit=self.window)]
        window_games=[_ for _ in window_games if _]

        # get opponent stats
        query={"teams": self.oid, "date": {
            "$lte": self.end_date, "$gte": self.season_start}}
        window_games_opp=[self.parseGameRow(g) for g in nba_games_collection.find(
            query, sort=[("date", -1)], limit=self.window)]
        window_games_opp=[_ for _ in window_games_opp if _]

        if not window_games:
            raise Exception("No previous games found for team %s, date %s" % (
                self.tid, self.end_date))
        if not window_games_opp:
            raise Exception("No previous games found for opp team %s, date %s" % (
                self.oid, self.end_date))
        '''
        print "Found %s own rows" % len(window_games)
        print "Found %s opp rows" % len(window_games_opp)
        '''
        return (window_games, window_games_opp)

    def loadInput(self, samples=30, recalculate=False):
        # try to find in database
        saved_query={"date": self.end_date,
                       "team_id": self.tid, "window": self.window}
        saved=nba_team_vectors_collection.find_one(saved_query)
        if saved and not recalculate:
            self.input=self.deserialize(saved['input'])
            return

        self.input={}

        window_games, window_games_opp=self.getWindowGames()
        window_games=self.imputeRows(self.mergeRows(window_games, self.tid))
        window_games_opp=self.imputeRows(
            self.mergeRows(window_games_opp, self.oid))

        sample_means=self.bootstrap(window_games, samples, df_mean)
        sample_means_opp=self.bootstrap(window_games_opp, samples, df_mean)
        sample_vars=self.bootstrap(window_games, samples, df_var)
        sample_vars_opp=self.bootstrap(window_games_opp, samples, df_var)

        # calculate averages
        self.input['means']=sample_means
        self.input['means_opp']=sample_means_opp
        self.input['variances']=sample_vars
        self.input['variances_opp']=sample_vars_opp

        # parse game contextual features from game row and team row; location,
        # home/road
        if self.home_team == self.tid:
            self.input['home/road']='home'
            self.input['location']=self.team_row['TEAM_CITY']
        else:
            self.input['home/road']='road'
            self.input['location']=self.team_row_opp['TEAM_CITY']

        '''
        for k,v in self.input.iteritems():
            print k
            print type(v)
            if isinstance(v, dict):
                for a,b in v.iteritems():
                    print a
                    print b
            else:
                print v
        '''
        # save/update
        data={'input': self.serialize(copy.deepcopy(
            self.input)), 'team_id': self.tid, 'date': self.end_date, 'window': self.window}
        if saved:
            nba_conn.updateDocument(nba_team_vectors_collection, saved[
                                    '_id'], data, upsert=False)
        else:
            nba_conn.saveDocument(nba_team_vectors_collection, data)

    def loadOutput(self):
        # parse game row for outputs
        parsed=self.parseGameRow(self.game_row)
        if not parsed:
            print "Game Row could not be parsed: possible bad game or future game"
            self.output={}
        else:
            data=self.imputeRows(self.mergeRows([parsed], self.tid))
            self.output=data


class playerFeatureVector(featureVector):

    PLAYER_FRAMES=['PlayerTrack',
                     'PlayerStats',
                     'sqlPlayersUsage',
                     'sqlPlayersScoring',
                     'sqlPlayersFourFactors',
                     'sqlPlayersMisc']

    PER36_KEYS=['AST',
                  'BLK',
                  'BLKA',
                  'CFGA',
                  'CFGM',
                  'DFGA',
                  'DFGM',
                  'DIST',
                  'DRBC',
                  'DREB',
                  'FG3A',
                  'FG3M',
                  'FGA',
                  'FGM',
                  'FTA',
                  'FTAST',
                  'FTM',
                  'OPP_PTS_2ND_CHANCE',
                  'OPP_PTS_FB',
                  'OPP_PTS_OFF_TOV',
                  'OPP_PTS_PAINT',
                  'ORBC',
                  'OREB',
                  'PASS',
                  'PF',
                  'PFD',
                  'PTS',
                  'PTS_2ND_CHANCE',
                  'PTS_FB',
                  'PTS_OFF_TOV',
                  'PTS_PAINT',
                  'RBC',
                  'REB',
                  'SAST',
                  'STL',
                  'TCHS',
                  'TO',
                  'UFGA',
                  'UFGM']

    def __init__(self, pid, tid, gid, window=10, recalculate=False):
        self.pid=int(pid)
        self.gid=str(gid)
        self.tid=int(tid)
        self.window=int(window)
        self.long_window=self.window * 2

        # get invalids, game info
        self.game_row=self.getGameRow(self.gid)
        self.invalids=self.game_row['InactivePlayers']
        self.gamesummary=self.game_row['GameSummary'].iloc[0]
        self.home_team=int(self.gamesummary['HOME_TEAM_ID'])
        self.away_team=int(self.gamesummary['VISITOR_TEAM_ID'])
        if self.tid not in set([self.home_team, self.away_team]):
            raise Exception("Team %s not in game %s" % (self.tid, self.gid))
        self.oid=self.home_team if self.home_team == self.tid else self.away_team

        # get team vectors
        self.own_team_vector=teamFeatureVector(
            self.tid, self.gid, window=self.window, recalculate=False)
        self.opp_team_vector=teamFeatureVector(
            self.oid, self.gid, window=self.window, recalculate=False)

        print "Obtained Team Vectors"

        self.output=None
        self.loadOutput()

        print "Loaded Output"

        self.input=None
        self.loadInput(samples=30, recalculate=recalculate)

        print "Loaded Input"

    def getGamePosition(self):
        '''
        Get the player's position in the specified game
        '''
        if self.output is None:
            # TODO: specify contingency for future games
            print "Can't find position in output, going to DB"
            playerpos=self.getPlayerPosition(pid)
        else:
            player_pos=self.output.iloc[0]['START_POSITION'].strip()
            if player_pos == '':
                player_pos=self.getPlayerPosition(pid)
            else:
                player_pos=[player_pos]
        return player_pos

    def parseGameRow(self, g, check_player=False):
        parsed={}
        success=True
        gid=str(g['_id'])
        parsed['date']=g['date']
        parsed['teams']=g['teams']
        parsed['_id']=gid
        # pull out home and away
        summary=g["GameSummary"]
        if isinstance(summary, str) or isinstance(summary, unicode):
            summary=pd.read_json(summary)
        parsed['home_team_id']=int(summary['HOME_TEAM_ID'][0])
        parsed['visitor_team_id']=int(summary['VISITOR_TEAM_ID'][0])
        for k, df in g.iteritems():
            if k in self.PLAYER_FRAMES:
                if isinstance(df, str) or isinstance(df, unicode):
                    df=pd.read_json(df)

                # check for empty
                if len(df.index) == 0:
                    success=False
                    break

                # check for advanced stats
                if k == 'PlayerStats':
                    if 'OFF_RATING' not in df:
                        success=False
                        break

                # check if player id is in here
                if check_player and len(df[df['PLAYER_ID'] == self.pid].index) == 1:
                    print "PLAYER IN GAME STATS %s for %s" % (k, gid)

                # sometimes tracking is empty, if DIST is 0, then set other
                # zero tracking stats to nan
                if k == "PlayerTrack":
                    zero_columns=[c for c in df.columns if df[c].sum() == 0]
                    if 'DIST' in zero_columns:
                        for c in zero_columns:
                            df[c]=np.nan

                # add game id and date
                if 'GAME_ID' not in df:
                    df['GAME_ID']=int(gid)
                df['GAME_DATE']=g['date']
                # make sure player id is an integer
                df['PLAYER_ID']=df['PLAYER_ID'].astype(int)

                parsed[k]=df
        if success:
            return parsed
        return None

    def mergeRows(self, rows, pid):
        merged={}
        for k in self.PLAYER_FRAMES:
            all_rows=pd.concat([_[k] for _ in rows])
            merged[k]=all_rows[all_rows['PLAYER_ID'] == pid]
        return merged

    def getWindowGames(self):
        '''
        get opponent team vectors for the windowed games as well (for strength of schedule)
        '''
        # get own stats
        query={"teams": self.tid, "date": {
            "$lte": self.own_team_vector.end_date, "$gte": self.own_team_vector.season_start}}
        window_games=[self.parseGameRow(g) for g in nba_games_collection.find(
            query, sort=[("date", -1)], limit=self.window)]
        window_games=[_ for _ in window_games if _]
        return window_games

    def imputeRows(self, row, join_shot_charts=True):
        '''
        Make necessary conversions/calcutions:
        - Fill in MIN=0.0
        - Convert to per36
        - Remove zero minute games
        - Remove blacklisted columns
        '''
        col_blacklist=['COMMENT',
                       'TEAM_ABBREVIATION',
                       'TEAM_CITY']
        # merge frames
        merged=None
        for k, v in row.iteritems():
            if k in self.PLAYER_FRAMES:
                if merged is None:
                    merged=v
                    continue
                else:
                    cols_to_use=(v.columns - merged.columns).tolist()
                    cols_to_use.append('GAME_ID')
                    merged=pd.merge(merged, v[cols_to_use], on='GAME_ID')
        if 'MIN' in merged:
            merged['MIN']=merged['MIN'].fillna(value='0:0').str.split(
                ':').apply(lambda x: float(x[0]) + float(x[1]) / 60.0)

        # join shot charts
        shot_chart_rows = []
        if join_shot_charts:
            for i, row in merged.iterrows():
                query = {"game_id": str(row['GAME_ID']).zfill(10), "player_id": row['PLAYER_ID']}
                sc_for_game = self.aggregateShotChart(query)
                sc_for_game['GAME_ID'] = row['GAME_ID']
                shot_chart_rows.append(sc_for_game)
            shot_chart_df = pd.DataFrame(shot_chart_rows)
            cols_to_use = (shot_chart_df.columns-merged.columns).tolist()
            cols_to_use.append('GAME_ID')
            merged = pd.merge(merged, shot_chart_df[cols_to_use], on='GAME_ID')

        # convert stats to per36 min
        multipliers=36.0 / merged['MIN']
        for k in self.PER36_KEYS:
            if k in merged:
                merged[k]=merged[k] * multipliers

        # remove zero minute games
        merged=merged[merged['MIN'] > 0]

        # drop columns
        for c in col_blacklist:
            if c in merged:
                merged.drop(c, axis=1, inplace=True)

        return merged

    def loadSplits(self, samples, days_rest, recalculate=False):
        '''
        hardcoded for up to 100 game window
        '''
        saved_query={"date": self.own_team_vector.end_date,
                       "player_id": self.pid}
        saved=nba_split_vectors_collection.find_one(saved_query)
        if saved and not recalculate:
            return self.deserialize(saved['input'])

        # calculate splits
        splits={'home/road': [],
                  'against_team': []}
        query={"players": self.pid, "date": {
            "$lte": self.own_team_vector.end_date}}
        window_games=[self.parseGameRow(g) for g in nba_games_collection.find(
            query, sort=[("date", -1)], limit=200)]
        window_games=[_ for _ in window_games if _]
        all_keys=set()
        home_game_keys=set()
        road_game_keys=set()
        against_team_keys=set()
        days_rest_keys=set()
        for i, w in enumerate(window_games):
            gid=int(w['_id'])
            all_keys.add(gid)
            # find which team player was on
            stats=w['PlayerStats']
            player_team_id=int(
                stats[stats['PLAYER_ID'] == self.pid]['TEAM_ID'])
            # find if home or away
            visitor=int(w['visitor_team_id'])
            home=int(w['home_team_id'])
            opp_id=None
            if home == player_team_id:
                home_game_keys.add(gid)
                opp_id=visitor
            elif visitor == player_team_id:
                road_game_keys.add(gid)
                opp_id=home
            else:
                raise Exception("Split ERROR")
            # find days of rest
            '''
            if i < len(window_games)-1:
                if (w['date'] - window_games[i+1]['date']).days == days_rest:
                    days_rest_keys.add(gid)
            '''
            # find if game opponent matches current opponent
            if opp_id == self.oid:
                against_team_keys.add(gid)

        print home_game_keys
        print road_game_keys
        print against_team_keys

        # gather rows
        merged=self.imputeRows(self.mergeRows(window_games, self.pid))
        merged=merged.set_index(
            ['GAME_ID'], drop=False, append=False, inplace=False, verify_integrity=False)
        home_stats=merged.ix[list(home_game_keys)]
        road_stats=merged.ix[list(road_game_keys)]
        against_team_stats=merged.ix[list(against_team_keys)]
        total_stats=merged.ix[list(all_keys)]

        # bootstrap means
        total_mean=self.bootstrap(total_stats, samples, df_mean)
        if len(home_stats.index) > 0:
            home_mean=self.bootstrap(home_stats, samples, df_mean)
        else:
            home_mean=total_mean
        if len(road_stats.index) > 0:
            road_mean=self.bootstrap(road_stats, samples, df_mean)
        else:
            road_mean=total_mean
        if len(against_team_stats.index) > 0:
            against_team_mean=self.bootstrap(
                against_team_stats, samples, df_mean)
        else:
            against_team_mean=total_mean

        # difference from total mean (for home mean, road mean, and
        # against_team mean)
        home_mean_diff=home_mean - total_mean
        road_mean_diff=road_mean - total_mean
        against_team_mean_diff=against_team_mean - total_mean

        print home_mean_diff
        print road_mean_diff
        print against_team_mean_diff

        input_dict={'home_split': home_mean_diff,
                      'road_split': road_mean_diff,
                      'against_team_split': against_team_mean_diff}

        # save
        to_save={"input": self.serialize(input_dict),
                   "date": self.own_team_vector.end_date,
                   "player_id": self.pid}
        if saved:
            nba_conn.updateDocument(nba_split_vectors_collection, saved[
                                    '_id'], to_save, upsert=False)
        else:
            nba_conn.saveDocument(nba_split_vectors_collection, to_save)

        return input_dict

    def loadTrends(self, player_data, pos, against_data, samples):
        '''
        calculate how the player has performed against average performance against opponent
        '''
        trends={}
        p_data = player_data.copy()
        p_data = p_data.convert_objects(convert_numeric=True)
        p_data = p_data._get_numeric_data().astype('float64')
        for p in pos:
            pos_trends=[]
            for (gid, tid), against_row in against_data.iteritems():
                mean=against_row[p]['mean'].ix['mean'].astype('float64')
                var=against_row[p]['var'].ix['mean'].astype('float64')
                row=p_data[p_data['GAME_ID'] == int(gid)]
                if len(row.index) == 0:
                    print "NO PLAYER ROW FOR GAME %s" % gid
                    continue
                player_row=row.iloc[0]
                trend=(player_row - mean) / np.sqrt(var)
                trend[np.isinf(trend)] = np.nan
                print trend
                pos_trends.append(trend)
            alltrends=pd.concat(pos_trends, axis=1).transpose()
            mean_distr=self.bayesianBootstrap(
                alltrends, samples, df_mean, samplesize=samples)
            var_distr=self.bayesianBootstrap(
                alltrends, samples, df_var, samplesize=samples)
            trends[p]={'mean': mean_distr, 'var': var_distr}
        return trends

    def performanceAgainstTeam(self, opp_args, samples, recalculate=False):
        '''
        Get from DB or recalculate average performance against opposition (bayesian bootstrap)
        To define a neighborhood: sample position players who played more than 10 min in game
        '''
        data={}
        # get stats ubset (normalized) from players who have played similar
        # position against each team
        for gid, tid in opp_args:
            # look in db
            saved_query={"game_id": gid, "team_id": tid,
                           "window": self.long_window}
            saved=nba_against_vectors_collection.find_one(saved_query)
            if saved and not recalculate:
                data[(gid, tid)]=self.deserialize(saved['input'])
                continue

            # find opposition games
            vector=teamFeatureVector(
                tid, gid, window=self.window, recalculate=False)
            query={"teams": tid, "date": {
                "$lte": vector.end_date, "$gte": vector.season_start}}
            window_games=[self.parseGameRow(_, check_player=False) for _ in nba_games_collection.find(
                query, sort=[("date", -1)], limit=self.long_window)]
            window_games=[_ for _ in window_games if _]

            # find opposition players + positions
            opp_players={}
            for g in window_games:
                for i, r in g['PlayerStats'].iterrows():
                    if r['TEAM_ID'] == vector.tid:  # find opposition only
                        continue
                    if r['PLAYER_ID'] in opp_players:  # already found
                        continue
                    pid=r['PLAYER_ID']
                    pos=r['START_POSITION']
                    if pos == '':
                        try:
                            pos=self.getPlayerPosition(pid)
                        except Exception as e:
                            print e
                            continue
                    else:
                        pos=[pos]
                    opp_players[pid]=pos

            # collect opposition player rows
            by_position={'G': [],
                           'F': [],
                           'C': []}
            for pid, pos in opp_players.iteritems():
                rows=self.imputeRows(self.mergeRows(window_games, pid))
                for p in pos:
                    by_position[p].append(rows.copy())

            # join into dataframes
            for k, v in by_position.iteritems():
                allrows=pd.concat(v, axis=0).reset_index(drop=True)
                print "%s: %s" % (k, allrows.shape)
                mean_distr=self.bayesianBootstrap(
                    allrows, samples, df_mean, samplesize=samples)
                var_distr=self.bayesianBootstrap(
                    allrows, samples, df_var, samplesize=samples)
                by_position[k]={"mean": mean_distr, "var": var_distr}
            data[(gid, tid)]=by_position

            # save
            to_save={"input": self.serialize(copy.deepcopy(
                by_position)), "game_id": gid, "team_id": tid, "window": self.long_window}
            if saved:
                nba_conn.updateDocument(nba_against_vectors_collection, saved[
                                        '_id'], to_save, upsert=False)
            else:
                nba_conn.saveDocument(nba_against_vectors_collection, to_save)

        return data

    def loadInput(self, samples=30, recalculate=False):
        # try to find in database
        saved_query={"date": self.own_team_vector.end_date, "player_id": self.pid,
                       "game_id": self.gid, "team_id": self.tid, "window": self.window}
        saved=nba_player_vectors_collection.find_one(saved_query)
        if saved and not recalculate:
            self.input=self.deserialize(saved['input'])
            return

        print "Calculating INPUT"
        self.input={}

        # parse own stats
        window_games=self.getWindowGames()
        print "Window Games: %s" % len(window_games)
        most_recent=window_games[0]['date']
        opp_args=[(self.gid, self.own_team_vector.oid)]
        for _ in window_games:
            opp_team=_['teams'][0] if (
                _['teams'][0] != self.own_team_vector.tid) else _['teams'][1]
            opp_args.append((_['_id'], int(opp_team)))
        merged=self.mergeRows(window_games, self.pid)
        data=self.imputeRows(merged)

        # calculate averages
        expmean=self.exponentiallyWeightedMean(data)
        expvar=self.exponentiallyWeightedVar(data)
        sample_means=self.bootstrap(data, samples, df_mean)
        sample_vars=self.bootstrap(data, samples, df_var)
        self.input['means']=sample_means
        self.input['variances']=sample_vars
        self.input['expmean']=expmean
        self.input['expvar']=expvar

        # parse contextual stats (minutes played, days rest, etc)
        days_rest=(self.own_team_vector.game_row['date'] - most_recent).days
        print "Days Rest: %s" % days_rest
        self.input['days_rest']=days_rest

        # pull up relevant team vector features
        self.input['home/road']=self.own_team_vector.input['home/road']
        self.input['location']=self.own_team_vector.input['location']

        positions=self.getGamePosition()
        self.input['position']=positions

        # calculate splits
        print "loading splits"
        splits=self.loadSplits(samples, days_rest, recalculate=False)
        self.input['splits']=splits

        # calculate opposition strength
        print "loading against"
        against_data=self.performanceAgainstTeam(
            opp_args, samples, recalculate=False)
        against_row=against_data[(self.gid, self.own_team_vector.oid)]
        self.input['against']={_: against_row[_] for _ in positions}

        # calculate trends
        print "loading trends"
        trends=self.loadTrends(data, positions, against_data, samples)
        self.input['trend']=trends

        # save/update
        data={'input': self.serialize(copy.deepcopy(self.input)),
                'player_id': self.pid,
                'game_id': self.gid,
                'team_id': self.tid,
                'date': self.own_team_vector.end_date,
                'window': self.window}
        if saved:
            nba_conn.updateDocument(nba_player_vectors_collection, saved[
                                    '_id'], data, upsert=False)
        else:
            nba_conn.saveDocument(nba_player_vectors_collection, data)

    def loadOutput(self):
        '''
        parse game row for player performance (for historical games only)
        '''
        # parse game row    for outputs
        parsed=self.parseGameRow(self.own_team_vector.game_row)
        if not parsed:
            print "Game Row could not be parsed: possible bad game or future game"
            self.output=None
        else:
            data=self.imputeRows(self.mergeRows([parsed], self.pid))
            self.output=data


if __name__ == '__main__':
    '''
    gid="0021400585"
    tid="1610612749"
    pid="201564"
    '''

    gid="0021401069"
    tid=1610612755
    pid=202397

    '''
    v = teamFeatureVector(tid, gid, window=10, recalculate=True)
    print list(v.input['means'].columns)
    '''

    v=playerFeatureVector(pid, tid, gid, window=10, recalculate=True)
    for k, value in v.input.iteritems():
        print k
        print value
