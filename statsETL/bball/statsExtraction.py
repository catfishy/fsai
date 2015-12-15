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
import time
from io import StringIO

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from pymongo.helpers import DuplicateKeyError
from celery import Celery, group

from statsETL.db.mongolib import *

WINDOW = 10 # TODO: LET THIS BE VARIED

def getGamesForDay(date):
    date = datetime(day=date.day, month=date.month, year=date.year)
    query = {"date" : {"$gte" : date, "$lt": date+timedelta(days=1)}}
    games = nba_games_collection.find(query)
    return games

def getTeamsForDay(date):
    games = getGamesForDay(date)
    args = []
    for g in games:
        gid = g['_id']
        for tid in g['teams']:
            args.append((int(tid), gid))
    return args

def getPlayersForDay(date):
    games = getGamesForDay(date)
    args = []
    for g in games:
        gid = g['_id']
        for pid, tid in g['player_teams'].iteritems():
            args.append((int(pid), int(tid), gid))
    return args

def getPlayerVector(pid, tid, gid, recalculate=False, window=None):
    if not window:
        window = WINDOW
    vector = playerFeatureVector(
        pid, tid, gid, window=window, recalculate=recalculate)
    return vector


def getTeamVector(tid, gid, recalculate=False, window=None):
    if not window:
        window = WINDOW
    vector = teamFeatureVector(tid, gid, window=window, recalculate=recalculate)
    return vector


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

def getPlayerOutputTrend(pid, gid, tid, window):
    '''
    Shortcut to get output trends without actually loading the vector object
    '''
    # look in database
    query = {"game_id": gid, "player_id": pid, "window": window}
    result = nba_conn.findOne(nba_player_outputs_collection, query=query)
    data_df = pd.DataFrame()
    if result is not None:
        data = result['input']
        if data is not None:
            data_df = pd.read_csv(StringIO(data)).set_index('id')
    else:
        player_vector = playerFeatureVector(pid, tid, gid, minimal=True)
        if player_vector.output is not None:
            data_df = player_vector.output
    output_row = data_df.loc['output'] if 'output' in data_df.index else None
    trend_row = data_df.loc['trend'] if 'trend' in data_df.index else None
    if output_row is None or trend_row is None:
        print "Can't get output/trend for %s, %s" % (pid, gid)
    return (output_row, trend_row)


class featureVector():

    def __init__(self):
        pass

    def saveOrUpdate(self, coll, query, input_to_save):
        input_to_save = self.serialize(input_to_save)
        try:
            to_save = copy.deepcopy(query)
            to_save['input'] = input_to_save
            nba_conn.saveDocument(coll, to_save)
        except DuplicateKeyError as e:
            saved = coll.find_one(query)
            nba_conn.updateDocument(coll, saved['_id'], {'input': input_to_save}, upsert=False)       

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
        pos=[str(_) for _ in set(pos)]
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
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.to_csv()
        elif isinstance(data, dict):
            data = {k: self.serialize(v) for k,v in data.iteritems()}
        elif isinstance(data, list):
            data = [self.serialize(_) for _ in data]
        else:
            data = copy.deepcopy(data)
        return data

    def translateGameCode(self):
        gamecode = self.gamesummary['GAMECODE'].split('/')[-1]
        home = self.gamesummary['HOME_TEAM_ID']
        away = self.gamesummary['VISITOR_TEAM_ID']
        away_abbr = gamecode[:3]
        home_abbr = gamecode[3:]
        newcode = '%s@%s' % (away_abbr,home_abbr)
        abbr = away_abbr if self.tid == away else home_abbr
        return newcode, abbr

    def deserialize(self, data):
        if isinstance(data, dict):
            data = {k: self.deserialize(v) for k,v in data.iteritems()}
        elif isinstance(data, list):
            data = [self.deserialize(_) for _ in data]
        elif data is None:
            pass
        else:
            try:
                data = pd.read_csv(StringIO(data)).set_index('id')
            except Exception as e:
                print "Deserialize problem: %s, %s" % (data,e)
        return data

    def exponentiallyWeightedMean(self, data):
        df=data.copy()
        df=df.convert_objects(convert_numeric=True)
        # set game date as index
        df=df.set_index(['GAME_DATE'], drop=False, append=False, inplace=False, verify_integrity=False)
        df=df.sort_index(axis=0)
        df=df._get_numeric_data()
        result=pd.ewma(df, span=len(df.index), adjust=True, ignore_na=True)
        result_row=result.tail(1)
        return result_row

    def exponentiallyWeightedVar(self, data):
        df=data.copy()
        df=df.convert_objects(convert_numeric=True)
        # set game date as index
        df=df.set_index(['GAME_DATE'], drop=False, append=False, inplace=False, verify_integrity=False)
        df=df.sort_index(axis=0)
        df=df._get_numeric_data()
        result=result=pd.ewmvar(df, span=len(df.index), adjust=True, ignore_na=True)
        result_row=result.tail(1)
        return result_row

    def bootstrap(self, data, num_samples, statistic):
        """
        Returns the results from num_samples bootstrap samples
        """
        # Generate the indices for the required number of
        # permutations/(resamplings with replacement) required
        data=data.copy()
        data=data.convert_objects(convert_numeric=True)
        data=data._get_numeric_data()
        #print "Bootstrap: %s, %s" % (statistic, data.shape)
        idx=np.random.randint(0, len(data.index), (num_samples, len(data)))
        stats=[statistic(data.iloc[idx_row]) for idx_row in idx]
        # Generate the multiple resampled data set from the original one
        samples=pd.concat(stats, axis=1).transpose()
        description=samples.describe()
        mean_df = pd.DataFrame(description.ix['mean']).T
        mean_df.reset_index(drop=True,inplace=True)
        return mean_df

    def bayesianBootstrap(self, data, num_samples, statistic, samplesize):
        def Dirichlet_sample(m, n):
            """Returns a matrix of values drawn from a Dirichlet distribution with parameters = 1.
            'm' rows of values, with 'n' Dirichlet draws in each one."""
            # Draw from Gamma distribution
            # Set Dirichlet distribution parameters
            Dirichlet_params=np.ones(m * n)
            # https://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution
            Dirichlet_weights=np.asarray([random.gammavariate(a, 1) for a in Dirichlet_params])
            Dirichlet_weights=Dirichlet_weights.reshape(m, n)  # Fold them (row by row) into a matrix
            row_sums=Dirichlet_weights.sum(axis=1)
            # Reweight each row to be normalised to 1
            Dirichlet_weights=Dirichlet_weights / row_sums[:, np.newaxis]
            return Dirichlet_weights

        data=data.copy()
        data=data.convert_objects(convert_numeric=True)
        data=data._get_numeric_data()
        #print "Bayesian Bootstrap: %s, %s" % (statistic, data.shape)
        # Generate sample of Dirichlet weights
        Dirich_wgts_matrix=Dirichlet_sample(num_samples, data.shape[0])

        # If statistic can be directly computed using the weights (such as the mean), do this since it will be faster.
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
        description = results.describe()
        mean_df = pd.DataFrame(description.ix['mean']).T
        mean_df.reset_index(drop=True,inplace=True)
        return mean_df


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
                 'OtherStats']

    def __init__(self, tid, gid, window=10, recalculate=False, no_stats=False, minimal=False):
        '''
        If minimal, only save arguments and load game row
        If no_stats, save contextual fields, but don't calculate input/output/season avgs
        '''
        self.tid=int(tid)
        self.gid=str(gid).zfill(10)
        self.window=int(window)

        # find the game
        self.game_row=self.getGameRow(self.gid)
        self.gamesummary=self.game_row['GameSummary'].iloc[0]

        # get dates
        self.end_date=min(datetime.now(), self.game_row['date'] - timedelta(days=1))
        start_year=self.end_date.year - 1 if (self.end_date < datetime(day=1, month=10, year=self.end_date.year)) else self.end_date.year
        self.season_start=datetime(day=1, month=10, year=start_year)

        if minimal:
            return
        
        self.home_team=int(self.gamesummary['HOME_TEAM_ID'])
        self.away_team=int(self.gamesummary['VISITOR_TEAM_ID'])
        if self.tid not in set([self.home_team, self.away_team]):
            raise Exception("Team %s not in game %s" % (self.tid, self.gid))
        self.oid=self.home_team if self.away_team == self.tid else self.away_team

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

        # find days rest
        self.days_rest = self.getDaysRest()

        # find location
        if self.home_team == self.tid:
            self.homeroad = 'home'
            self.location = self.team_row['TEAM_CITY']
        else:
            self.homeroad = 'road'
            self.location = self.team_row_opp['TEAM_CITY']

        if no_stats:
            return

        self.output = None
        self.season_stats=None
        self.input = None

        self.saved_query={"date": self.end_date, "team_id": self.tid, "window": self.window}
        self.season_saved_query={"date": self.end_date, "team_id": self.tid}
        self.output_saved_query={'team_id': self.tid, "game_id": self.gid, }
        
        try:
            self.loadOutput(recalculate=recalculate)
        except Exception as e:
            #print "Can't load team output: %s" % e
            self.saveOutput()
        try:
            self.loadInput(recalculate=recalculate)
        except Exception as e:
            #print "Can't load team input: %s" % e
            self.saveInput()
        try:
            self.loadSeasonAverages(recalculate=recalculate)
        except Exception as e:
            #print "Can't load team season avgs: %s" % e
            self.saveSeasonAverages()

    def getDaysRest(self):
        query={"teams": self.tid, "date": {"$lte": self.end_date, "$gte": self.season_start}}
        games = [_ for _ in nba_games_collection.find(query, sort=[("date", -1)], limit=1)]
        if games:
            most_recent = games[0]['date']
            days_rest = (self.game_row['date'] - most_recent).days
            days_rest = min(10, days_rest)
        else:
            days_rest = 10
        return days_rest

    def loadSeasonAverages(self, samples=30, recalculate=False):
        '''
        Get team averages over a longer window (bootstrapped)
        '''
        # try to find in database
        saved = nba_season_averages_collection.find_one(self.season_saved_query)
        if saved and not recalculate:
            self.season_stats = self.deserialize(saved['input'])
            return

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
        season_means['id'] = 'means'
        season_vars=self.bayesianBootstrap(
            season_games, samples, df_var, samplesize=samples)
        season_vars['id'] = 'vars'
        
        season_stats_df = pd.concat((season_means, season_vars))
        season_stats_df.set_index('id', inplace=True)

        self.season_stats = season_stats_df
        self.saveSeasonAverages()

    def saveSeasonAverages(self):
        self.saveOrUpdate(nba_season_averages_collection, 
                          self.season_saved_query,
                          self.season_stats)

    def mergeRows(self, rows, tid):
        merged={}
        for k in self.TEAM_FRAMES:
            all_rows=pd.concat([_[k] for _ in rows])
            merged[k]=all_rows[all_rows['TEAM_ID'] == tid]
        return merged

    def imputeRows(self, row, join_shot_charts=True):
        col_blacklist=['LEAGUE_ID', 'MIN']

        # merge frames
        merged=None
        for k, v in row.iteritems():
            if k in self.TEAM_FRAMES:
                if merged is None:
                    merged=v
                    continue
                else:
                    cols_to_use=v.columns.difference(merged.columns).tolist()
                    cols_to_use.append('GAME_ID')
                    merged=pd.merge(merged, v[cols_to_use], on='GAME_ID')

        # join shot charts
        shot_chart_rows = []
        if join_shot_charts and len(merged.index) > 0:
            for i, row in merged.iterrows():
                query = {"game_id": str(row['GAME_ID']).zfill(10), "TEAM_ID": row['TEAM_ID']}
                sc_for_game = self.aggregateShotChart(query)
                sc_for_game['GAME_ID'] = row['GAME_ID']
                shot_chart_rows.append(sc_for_game)
            shot_chart_df = pd.DataFrame(shot_chart_rows)
            cols_to_use = shot_chart_df.columns.difference(merged.columns).tolist()
            cols_to_use.append('GAME_ID')
            merged = pd.merge(merged, shot_chart_df[cols_to_use], on='GAME_ID')

        if 'MIN' in merged:
            merged['MIN']=merged['MIN'].fillna(value='0:0').str.split(':').apply(lambda x: float(x[0]) + float(x[1]) / 60.0)

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

        # get own stats
        query={"teams": self.tid, "date": {
            "$lte": self.end_date, "$gte": self.season_start}}
        window_games=[self.parseGameRow(g) for g in nba_games_collection.find(
            query, sort=[("date", -1)], limit=self.window)]
        window_games=[_ for _ in window_games if _]

        if not window_games:
            raise Exception("No previous games found for team %s, date %s" % (
                self.tid, self.end_date))

        return window_games

    def loadInput(self, samples=30, recalculate=False):
        # try to find in database
        saved=nba_team_vectors_collection.find_one(self.saved_query)
        if saved and not recalculate:
            self.input = self.deserialize(saved['input'])
            return

        window_games = self.getWindowGames()
        window_games=self.imputeRows(self.mergeRows(window_games, self.tid))
        sample_means=self.bootstrap(window_games, samples, df_mean)
        sample_means['id'] = 'means'
        sample_vars=self.bootstrap(window_games, samples, df_var)
        sample_vars['id'] = 'vars'
        sample_df = pd.concat((sample_means, sample_vars))
        sample_df.set_index('id', inplace=True)
        self.input = sample_df

        self.saveInput()

    def saveInput(self):
        self.saveOrUpdate(nba_team_vectors_collection, 
                          self.saved_query,
                          self.input)

    def loadOutput(self, recalculate=False):
        # try to find in database
        saved = nba_team_outputs_collection.find_one(self.output_saved_query)
        if saved and not recalculate:
            self.output = self.deserialize(saved['input'])
            return

        # parse game row for outputs
        parsed=self.parseGameRow(self.game_row)
        if not parsed:
            print "No output, future? (tid %s, gid %s)" % (self.tid, self.gid)
        else:
            data = self.imputeRows(self.mergeRows([parsed], self.tid))
            data = data.head(1)
            data['id'] = 'output'
            data.set_index('id', inplace=True)
            self.output=data

        self.saveOutput()

    def saveOutput(self):
        self.saveOrUpdate(nba_team_outputs_collection, 
                          self.output_saved_query,
                          self.output)


class playerFeatureVector(featureVector):

    '''
    TODO:
    - Include pace adjusted stats (have to get # possessions first)
    '''

    PLAYER_FRAMES=['PlayerTrack',
                     'PlayerStats',
                     'sqlPlayersUsage',
                     'sqlPlayersScoring',
                     'sqlPlayersFourFactors',
                     'sqlPlayersMisc']

    PER36_KEYS=['AST','BLK','BLKA','CFGA','CFGM','DFGA','DFGM','DIST','DRBC','DREB','FG3A','FG3M','FGA','FGM','FTA',
                'FTAST','FTM','OPP_PTS_2ND_CHANCE','OPP_PTS_FB','OPP_PTS_OFF_TOV','OPP_PTS_PAINT','ORBC','OREB','PASS',
                'PF','PFD','PTS','PTS_2ND_CHANCE','PTS_FB','PTS_OFF_TOV','PTS_PAINT','RBC','REB','SAST','STL','TCHS',
                'TO','UFGA','UFGM']
    PER36_KEYS+=['Above_the_Break_3_Back_Court(BC)_Back_Court_Shot_attempted',
                'Above_the_Break_3_Center(C)_24+_ft_attempted',
                'Above_the_Break_3_Left_Side_Center(LC)_24+_ft_attempted',
                'Above_the_Break_3_Right_Side_Center(RC)_24+_ft_attempted',
                'Backcourt_Back_Court(BC)_Back_Court_Shot_attempted',
                'In_The_Paint_(Non_RA)_Center(C)_8_16_ft_attempted',
                'In_The_Paint_(Non_RA)_Center(C)_Less_Than_8_ft_attempted',
                'In_The_Paint_(Non_RA)_Left_Side(L)_8_16_ft_attempted',
                'In_The_Paint_(Non_RA)_Right_Side(R)_8_16_ft_attempted',
                'Left_Corner_3_Left_Side(L)_24+_ft_attempted',
                'Mid_Range_Center(C)_16_24_ft_attempted',
                'Mid_Range_Center(C)_8_16_ft_attempted',
                'Mid_Range_Left_Side(L)_16_24_ft_attempted',
                'Mid_Range_Left_Side(L)_8_16_ft_attempted',
                'Mid_Range_Left_Side_Center(LC)_16_24_ft_attempted',
                'Mid_Range_Right_Side(R)_16_24_ft_attempted',
                'Mid_Range_Right_Side(R)_8_16_ft_attempted',
                'Mid_Range_Right_Side_Center(RC)_16_24_ft_attempted',
                'Restricted_Area_Center(C)_Less_Than_8_ft_attempted',
                'Right_Corner_3_Right_Side(R)_24+_ft_attempted']

    def __init__(self, pid, tid, gid, window=10, recalculate=False, minimal=False):
        '''
        If minimal, only load output and mean/var/expmean/expvar inputs
        '''
        self.pid=int(pid)
        self.gid=str(gid).zfill(10)
        self.tid=int(tid)
        self.minimal = minimal
        self.window=int(window)
        self.long_window=self.window * 2
        self.getPlayerName()

        # get team vector 
        self.own_team_vector = teamFeatureVector(self.tid, self.gid, window=self.window, 
                                                 recalculate=False, no_stats=True, minimal=self.minimal)
        self.gamesummary = self.own_team_vector.gamesummary
        
        self.input={}
        self.output=None

        self.saved_query={"date": self.own_team_vector.end_date, "player_id": self.pid,
                          "game_id": self.gid, "team_id": self.tid, "window": self.window}
        self.output_saved_query={"game_id": self.gid, "player_id": self.pid, "window": self.window}
        self.splits_saved_query={"date": self.own_team_vector.end_date, "player_id": self.pid}

        try:
            self.loadInput(samples=30, recalculate=recalculate)
        except Exception as e:
            print "Can't load player input: %s" % e
            #traceback.print_exc()
            self.saveInput()
        try:
            self.loadOutput(recalculate=recalculate)
        except Exception as e:
            print "Can't load player output: %s" % e
            #traceback.print_exc()
            self.saveOutput()

    def getPlayerName(self):
        player_row = nba_conn.findByID(nba_players_collection, self.pid)
        if player_row:
            self.player_name = player_row["DISPLAY_FIRST_LAST"]
        else:
            print "CAN'T FIND PLAYER ROW: %s" % self.pid
            self.player_name = "?"

    def getGamePosition(self):
        '''
        Get the player's position in the specified game
        '''
        if self.output is None:
            print "No output, going to DB"
            player_pos=self.getPlayerPosition(self.pid)
        else:
            player_pos=self.output['START_POSITION'].strip()
            if player_pos == '':
                player_pos=self.getPlayerPosition(self.pid)
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
                    if df['DIST'].sum() == 0:
                        zero_columns=[c for c in df.columns if df[c].sum() == 0]
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
        window_games=[self.parseGameRow(g) for g in nba_games_collection.find(query, sort=[("date", -1)], limit=self.window)]
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
                    cols_to_use=v.columns.difference(merged.columns).tolist()
                    cols_to_use.append('GAME_ID')
                    merged=pd.merge(merged, v[cols_to_use], on='GAME_ID')
        if 'MIN' in merged:
            merged['MIN']=merged['MIN'].fillna(value='0:0').str.split(':').apply(lambda x: float(x[0]) + float(x[1]) / 60.0)

        # join shot charts
        shot_chart_rows = []
        if join_shot_charts and len(merged.index) > 0:
            for i, row in merged.iterrows():
                query = {"game_id": str(row['GAME_ID']).zfill(10), "player_id": row['PLAYER_ID']}
                sc_for_game = self.aggregateShotChart(query)
                sc_for_game['GAME_ID'] = row['GAME_ID']
                shot_chart_rows.append(sc_for_game)
            shot_chart_df = pd.DataFrame(shot_chart_rows)
            cols_to_use = shot_chart_df.columns.difference(merged.columns).tolist()
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
        saved = nba_split_vectors_collection.find_one(self.splits_saved_query)
        if saved and not recalculate:
            return self.deserialize(saved['input'])

        # calculate splits
        query = {"players": self.pid, "date": {"$lte": self.own_team_vector.end_date}}
        window_games=[self.parseGameRow(g) for g in nba_games_collection.find(query, sort=[("date", -1)], limit=200)]
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
            if opp_id == self.own_team_vector.oid:
                against_team_keys.add(gid)

        # gather rows
        merged=self.imputeRows(self.mergeRows(window_games, self.pid))
        merged=merged.set_index(['GAME_ID'], drop=False, append=False, inplace=False, verify_integrity=False)
        home_stats=merged.ix[list(home_game_keys)]
        road_stats=merged.ix[list(road_game_keys)]
        against_team_stats=merged.ix[list(against_team_keys)]
        total_stats=merged.ix[list(all_keys)]

        # bootstrap means
        total_mean=self.bootstrap(total_stats, samples, df_mean).iloc[0]
        if len(home_stats.index) > 0:
            home_mean=self.bootstrap(home_stats, samples, df_mean).iloc[0]
        else:
            home_mean=total_mean
        if len(road_stats.index) > 0:
            road_mean=self.bootstrap(road_stats, samples, df_mean).iloc[0]
        else:
            road_mean=total_mean
        if len(against_team_stats.index) > 0:
            against_team_mean=self.bootstrap(against_team_stats, samples, df_mean).iloc[0]
        else:
            against_team_mean=total_mean

        # difference from total mean (for home mean, road mean, and against_team mean)
        home_mean_diff=pd.DataFrame(home_mean - total_mean).T
        home_mean_diff['id'] = 'home_split'
        road_mean_diff=pd.DataFrame(road_mean - total_mean).T
        road_mean_diff['id'] = 'road_split'
        against_team_mean_diff=pd.DataFrame(against_team_mean - total_mean).T
        against_team_mean_diff['id'] = 'against_team_split'
        split_df = pd.concat([home_mean_diff, road_mean_diff, against_team_mean_diff])
        split_df.set_index('id',inplace=True)

        # save and return
        self.saveOrUpdate(nba_split_vectors_collection, self.splits_saved_query, split_df)
        return split_df


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
                # skip current game (just in case)
                if int(gid) == int(self.gid):
                    continue

                against_row_df = against_row[p]
                if 'mean' not in against_row_df.index or 'var' not in against_row_df.index:
                    print "No against stats for %s (%s, %s), skipping" % (p, gid, tid)
                    continue

                mean = against_row_df.loc['mean'].astype('float64')
                var = against_row_df.loc['var'].astype('float64')
                row = p_data[p_data['GAME_ID'] == int(gid)]
                if len(row.index) == 0:
                    print "PLAYER DID NOT PLAY GAME %s" % gid
                    continue
                player_row=row.iloc[0]
                trend = player_row.subtract(mean)

                num_zeros = trend.ix[trend==0.0].index
                var_zeros = var.ix[var==0.0].index
                both_zeros = num_zeros & var_zeros

                # fill in data
                trend = trend.divide(np.sqrt(var))
                trend[both_zeros] = 0.0
                trend[np.isinf(trend)] = np.nan

                pos_trends.append(trend)

            if len(pos_trends) == 0:
                print "NO OPPORTUNITIES TO MEASURE TREND FOR POSITION %s" % p
                mean_distr = pd.DataFrame()
                var_distr = pd.DataFrame()
            else:
                alltrends=pd.concat(pos_trends, axis=1).transpose()
                mean_distr=self.bayesianBootstrap(alltrends, samples, df_mean, samplesize=samples)
                var_distr=self.bayesianBootstrap(alltrends, samples, df_var, samplesize=samples)
            mean_distr['id'] = 'mean'
            var_distr['id'] = 'var'
            trend_df = pd.concat([mean_distr,var_distr])
            trend_df.set_index('id', inplace=True)
            trends[p] = trend_df
        return trends

    def loadAgainst(self, opp_args, samples, recalculate=False):
        data = {}
        for gid, tid in opp_args:
            result = self.performanceAgainstTeam(gid, tid, samples, recalculate=recalculate)
            if result is not None:
                data[(gid,tid)] = result
        return data
        
    def performanceAgainstTeam(self, gid, tid, samples, recalculate=False):
        '''
        Average/Trend performance against opposition by position
        Use players who played more than 10 min in game

        For the most part, the game is in the past (window games end at current date), 
        and if stats are up to date, this only needs to be calculated once
        '''
        saved_query={"game_id": gid, "team_id": tid, "window": self.long_window}

        saved = nba_against_vectors_collection.find_one(saved_query)
        if saved and not recalculate:
            perf_against = self.deserialize(saved['input'])
            print "FOUND OPP ARG %s, %s" % (gid, tid)
            return perf_against

        # find opposition games
        try:
            vector=teamFeatureVector(tid, gid, window=self.window, recalculate=False)
        except Exception as e:
            print "Can't find performance against team %s @ %s: %s" % (tid, gid, e)
            print traceback.print_exc()
            return None

        query={"teams": tid, "date": {"$lte": vector.end_date, "$gte": vector.season_start}}
        window_games=[self.parseGameRow(_, check_player=False) for _ in nba_games_collection.find(query, sort=[("date", -1)], limit=self.long_window)]
        window_games=[_ for _ in window_games if _]

        # find opposition players + positions
        by_position={'G': pd.DataFrame(),
                     'F': pd.DataFrame(),
                     'C': pd.DataFrame()}
        trend_by_position={'G': pd.DataFrame(),
                           'F': pd.DataFrame(),
                           'C': pd.DataFrame()}
        opp_players=[]
        window_games_by_id = {}
        for g in window_games:
            opp_gid = g['_id']
            window_games_by_id[opp_gid] = g
            player_stats = g['PlayerStats']
            player_stats['MIN']=player_stats['MIN'].fillna(value='0:0').str.split(':').apply(lambda x: float(x[0]) + float(x[1]) / 60.0)
            player_stats = player_stats[player_stats.MIN >= 10.0]
            player_stats = player_stats[player_stats.TEAM_ID != vector.tid]
            for i, r in player_stats.iterrows():
                pid=r['PLAYER_ID']
                pos=str(r['START_POSITION'])
                if pos == '':
                    try:
                        pos=self.getPlayerPosition(pid)
                    except Exception as e:
                        print "Error getting position for %s: %s" % (pid, e)
                        continue
                else:
                    pos=[pos]
                opp_players.append((pos,pid,opp_gid,r['TEAM_ID']))

        # aggregate opposition player trends
        random.shuffle(opp_players)
        for i, arg in enumerate(opp_players):
            #print "AGAINST ARG %s/%s" % (i+1,len(opp_players))
            pos, opp_pid, opp_gid, opp_tid = arg
            output_row, trend_row = getPlayerOutputTrend(opp_pid, opp_gid, opp_tid, self.window)
            for p in pos:
                if trend_row is not None:
                    trend_by_position[p] = pd.concat((trend_by_position[p],pd.DataFrame(trend_row).T), axis=0).reset_index(drop=True)
                if output_row is not None:
                    by_position[p] = pd.concat((by_position[p],pd.DataFrame(output_row).T),axis=0).reset_index(drop=True)

        # join into dataframes
        for k in by_position.keys():
            v = by_position[k]
            v_trend = trend_by_position[k]
            if len(v.index) > 0:
                mean_distr=self.bayesianBootstrap(v, samples, df_mean, samplesize=samples)
                var_distr=self.bayesianBootstrap(v, samples, df_var, samplesize=samples)
            else:
                mean_distr = pd.DataFrame()
                var_distr = pd.DataFrame()
            if len(v_trend.index) > 0 :
                trend_mean_distr=self.bayesianBootstrap(v_trend, samples, df_mean, samplesize=samples)
                trend_var_distr=self.bayesianBootstrap(v_trend, samples, df_var, samplesize=samples)
            else:
                trend_mean_distr = pd.DataFrame()
                trend_var_distr = pd.DataFrame()

            mean_distr['id'] = 'mean'
            var_distr['id'] = 'var'
            trend_mean_distr['id'] = 'trend_mean'
            trend_var_distr['id'] = 'trend_var'

            by_position_df = pd.concat([mean_distr, var_distr, trend_mean_distr, trend_var_distr])
            by_position_df.set_index('id',inplace=True)
            by_position[k] = by_position_df

        # save and return
        self.saveOrUpdate(nba_against_vectors_collection, saved_query, by_position)
        return by_position

    def loadInput(self, samples=30, recalculate=False):
        '''
        First, try to find fields saved into database. Saved data can
        be in one of three states:
        1. empty (only if error in own stat calculation)
        2. own stats only
        3. own + trend stats (fully saved)
        
        If the saved fields are sufficient (i.e. minimal or empty), 
        and we're not recalculating, then return
        
        If we are recalculating, then delete remove saved data and rerun
        
        If we are not recalculating, keep the saved data, and fill in 
        other fields (against,splits => pulled from separate db)
        '''
        saved=nba_player_vectors_collection.find_one(self.saved_query)
        if saved and not recalculate:
            self.input = self.deserialize(saved['input'])
            if len(self.input) == 0 or self.minimal:
                return

        window_games = None
        data = None

        if 'own' not in self.input:
            # get window games
            window_games=self.getWindowGames()
            if len(window_games) == 0:
                raise Exception("No Window Games")

            # parse own stats
            merged=self.mergeRows(window_games, self.pid)
            data=self.imputeRows(merged)
            if len(data.index) == 0:
                raise Exception("No Valid games found for %s (didn't play any minutes out of %s games)" % (self.pid, len(window_games)))

            # calculate averages
            expmean=self.exponentiallyWeightedMean(data)
            expmean['id'] = 'expmean'
            expvar=self.exponentiallyWeightedVar(data)
            expvar['id'] = 'expvar'
            sample_means=self.bootstrap(data, samples, df_mean)
            sample_means['id'] = 'means'
            sample_vars=self.bootstrap(data, samples, df_var)
            sample_vars['id'] = 'vars'

            sample_df = pd.concat([sample_means, sample_vars, expmean, expvar])
            sample_df.set_index('id', inplace=True)
            self.input['own'] = sample_df

            # early stop
            if self.minimal:
                self.saveInput()
                return

        # parse contextual stats
        days_rest=self.own_team_vector.days_rest
        positions=self.getGamePosition()
        print positions
        self.input['days_rest']=days_rest
        self.input['home/road']=self.own_team_vector.homeroad
        self.input['location']=self.own_team_vector.location
        self.input['position']=positions

        # calculate splits
        if 'splits' not in self.input:
            print "LOADING SPLITS"
            self.input['splits']=self.loadSplits(samples, days_rest, recalculate=recalculate)

        # load against
        if 'against' not in self.input:
            print "LOADING AGAINST"
            cur_against_data = self.performanceAgainstTeam(self.gid, self.own_team_vector.oid, samples, recalculate=recalculate)
            self.input['against']={_: cur_against_data[_] for _ in positions}

        # calculate trends
        if 'trend' not in self.input:
            print "LOADING TREND"
            if window_games is None or data is None:
                # get window games (should not throw errors as own data already loaded)
                window_games=self.getWindowGames()
                merged=self.mergeRows(window_games, self.pid)
                data=self.imputeRows(merged)
            opp_args=[]
            for _ in window_games:
                opp_team = _['teams'][0] if (_['teams'][0] != self.own_team_vector.tid) else _['teams'][1]
                opp_args.append((_['_id'], int(opp_team)))
            past_against_data = self.loadAgainst(opp_args, samples, recalculate=False)
            self.input['trend']=self.loadTrends(data, positions, past_against_data, samples)

        self.saveInput()

    def saveInput(self):
        # save
        to_save_keys = ['trend', 'own']
        to_save = {_:self.input[_] for _ in to_save_keys if _ in self.input}
        self.saveOrUpdate(nba_player_vectors_collection, self.saved_query, to_save)

    def loadOutput(self, recalculate=False):
        # look in db
        saved = nba_player_outputs_collection.find_one(self.output_saved_query)
        if saved and not recalculate:
            self.output = self.deserialize(saved['input'])

        # generate output
        parsed=self.parseGameRow(self.own_team_vector.game_row)
        if not parsed:
            print "No output, future? (pid %s, tid %s, gid %s)" % (self.pid, self.tid, self.gid)
        else:
            data=self.imputeRows(self.mergeRows([parsed], self.pid))
            if len(data.index) > 0:
                self.output=data.head(1)
                self.output.loc[:,'id'] = 'output'
                self.output.set_index('id', inplace=True)
            else:
                print "Parsed game row but no player output"

        # generate output trend
        self.loadOutputTrend()

        # save
        self.saveOutput()

    def loadOutputTrend(self):
        if not isinstance(self.output, pd.DataFrame):
            print "Don't have output for output trend"
            return
        if 'own' not in self.input or not isinstance(self.input['own'], pd.DataFrame):
            print "Don't have own stats for output trend"
            return

        # get mean performance and output performance
        own_df = self.input['own']
        mean = own_df.loc['means']
        var = own_df.loc['vars']
        current = self.output.loc['output']
        
        # get numeric data
        mean = mean.convert_objects(convert_numeric=True)._get_numeric_data()
        current = current.convert_objects(convert_numeric=True)._get_numeric_data()
        trend = current.subtract(mean)

        # fill in data
        num_zeros = trend.ix[trend==0.0].index
        var_zeros = var.ix[var==0.0].index
        both_zeros = num_zeros & var_zeros
        trend = trend.divide(np.sqrt(var))
        trend[both_zeros] = 0.0
        trend[np.isinf(trend)] = np.nan

        # convert to trend dataframe
        trend = pd.DataFrame(trend).T
        trend['id'] = 'trend'
        trend.set_index('id', inplace=True)

        # add onto output df
        self.output = pd.concat((self.output,trend))

    def saveOutput(self):
        self.saveOrUpdate(nba_player_outputs_collection, self.output_saved_query, self.output)


if __name__ == '__main__':
    # pid, tid, gid = (202397, 1610612755, '0021401069')
    #pid, tid, gid = (203095, 1610612753, '0011400098')
    #pid, tid, gid = (204064, 1610612745, '0011400098')
    #pid, tid, gid = (2772, 1610612745, '0011400098')
    pid, tid, gid = (203900, 1610612751, '0021500264')

    # tvect = getTeamVector(tid, gid, recalculate=True)
    # print tvect.output
    # print tvect.season_stats
    # print tvect.input

    pvect = getPlayerVector(pid, tid, gid, recalculate=True)
    print pvect.output
    for k,v in pvect.input.iteritems():
        print k
        print v
    sys.exit(1)


    

