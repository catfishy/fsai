'''
Classes that take in player/team vector,
and generate API responses for them
Also takes care of caching the response
'''
import sys
from datetime import datetime, timedelta
import json
from collections import defaultdict
import copy
import traceback

import numpy as np
import pandas as pd
import pymongo
from pymongo.helpers import DuplicateKeyError

from statsETL.util.redislib import RedisTTLCache
from statsETL.db.mongolib import *


TEAM_REDIS_CACHE = RedisTTLCache('teamapi')
PLAYER_REDIS_CACHE = RedisTTLCache('playerapi')
PLAYER_TYPES_ALLOWED = ['windowed', 'opponent_pos', 'trend_pos', 'exponential', 'homeroadsplit', 'oppsplit', 'meta']
TEAM_TYPES_ALLOWED = ['windowed', 'meta', 'season']
REVERSED = ['TO',
            'TM_TOV_PCT',
            'DFGA',
            'DFG_PCT',
            'OPP_FTA_RATE',
            'OPP_PTS_FB',
            'OPP_EFG_PCT',
            'OPP_OREB_PCT',
            'OPP_PTS_PAINT',
            'OPP_PTS_2ND_CHANCE',
            'OPP_PTS_OFF_TOV',
            'DFGM',
            'PCT_TOV',
            'PCT_PF'];

'''
Specifies which stat categories to query for, given a request type
'''
PLAYER_STAT_TYPE = {'basic': ['exponential','meta'],
                    'matchup': ['exponential','opponent_pos','trend_pos','homeroadsplit','oppsplit']}
TEAM_STAT_TYPE = {'basic': ['windowed','meta'],
                  'matchup': ['windowed','season']}

'''
Specifies which stats to return, given a request type
'''
BASIC_STAT_TYPE_KEYS = [('gid','GAME'),
                        ('tid','TEAM'),
                        ('pid','PLAYER'),
                        ('home/road','HOME/ ROAD'),
                        ('positions','POS'),
                        ('days_rest','REST'),
                        ('MIN','MIN'),
                        ('USG_PCT','USG%'),
                        ('PIE','PIE'),
                        ('TCHS','TOUCH'),
                        ('PTS','PTS'),
                        ('REB','REB'),
                        ('AST','AST'),
                        ('PASS','PASS'),
                        ('TO','TO'),
                        ('STL','STL'),
                        ('BLK','BLK'),
                        ('BLKA','BLKA'),
                        ('REB_PCT','REB%'),
                        ('AST_RATIO','AST RATIO'),
                        ('FGA','FGA'),
                        ('TS_PCT','TS%'),
                        ('FG3A','FG3A'),
                        ('FG3_PCT','FG3%'),
                        ('FTA_RATE','FTA RATE'),
                        ('FT_PCT','FT%'),
                        ('PCT_FGA','%FGA'),
                        ('PCT_FGM','%FGM'),
                        ('PCT_FG3A','%FG3M'),
                        ('PCT_FG3M','%FG3M'),
                        ('PCT_FTA','%FTA'),
                        ('PCT_FTM','%FTM'),
                        ('PCT_REB','%REB'),
                        ('PCT_AST','%AST'),
                        ('PCT_TOV','%TO'),
                        ('PCT_STL','%STL'),
                        ('PCT_PFD','%PFD'),
                        ('PCT_BLK','%BLK'),
                        ('PCT_BLKA','%BLKA'),
                        ('PCT_PTS','%PTS')]
MATCHUP_STAT_TYPE_KEYS = []

class colorExtractor(object):

    '''
    gradient from red to green
    '''

    def __init__(self, vector_type, date=None):
        self.vector_type = vector_type
        self.colorRange = {}
        self.colorRangeDf = pd.DataFrame()
        if date is not None:
            self.loadColorRange(date)

    def updateColorRange(self, newvals):
        new_row = {}
        for k,v in newvals.iteritems():
            try:
                v = float(v)
            except Exception as e:
                continue # don't save non float values into color range
            new_row[k] = v
        self.colorRangeDf = pd.concat((self.colorRangeDf,pd.DataFrame([new_row])))
        self.colorRangeDf.reset_index(drop=True, inplace=True)
        if len(self.colorRangeDf.index) > 1000: # cap at 1000 rows
            self.colorRangeDf.drop(self.colorRangeDf.index[:1],inplace=True)

    def dateKey(self, date):
        return datetime(year=date.year, month=date.month, day=date.day)

    def saveColorRange(self, date):
        # average df
        d = self.colorRangeDf.describe()
        mean = d.loc['mean']
        std = d.loc['std']
        minvals = d.loc['min']
        maxvals = d.loc['max']
        lower = mean - (2*std)
        upper = mean + (2*std)
        # clamp if necessary
        nonneg_cols = minvals >= 0
        lower[nonneg_cols] = lower[nonneg_cols].apply(lambda x: max(x,0))
        nonpos_cols = maxvals <= 0
        upper[nonpos_cols] = upper[nonpos_cols].apply(lambda x: min(x,0))
        ranges = pd.concat((lower,upper),axis=1).T
        ranges.index = ['lower','upper']
        # convert to dict
        self.colorRange = {c: list(ranges.loc[:,c])  for c in ranges.columns}
        # save
        try:
            to_save = {"date": self.dateKey(date), "vector_type": self.vector_type, "data": self.colorRange}
            nba_conn.saveDocument(nba_stat_ranges_collection, to_save)
        except DuplicateKeyError as e:
            saved_query = {"date": self.dateKey(date), "vector_type": self.vector_type}
            saved = nba_stat_ranges_collection.find_one(saved_query)
            nba_conn.updateDocument(nba_stat_ranges_collection, saved['_id'], {'data': self.colorRange}, upsert=False)

    def loadColorRange(self, date):
        query = {'vector_type': self.vector_type, 
                 'date': {"$lte": self.dateKey(date)}}
        results = nba_stat_ranges_collection.find(query, sort=[('date',pymongo.DESCENDING)], limit=1)
        if results:
            self.colorRange = results[0]['data']
            return True
        return False

    def extractColor(self, key, value):
        # default transparent
        if key not in self.colorRange:
            return 'transparent'

        # determine reversal
        key_parts = key.split('_')
        prefix = key_parts[0]
        nonprefix_key = '_'.join(key_parts[1:])
        if nonprefix_key in REVERSED:
            reversed = True
        else:
            reversed = False

        # in case there is a whole stat type where ranks are flipped
        '''
        if 'opponent' == prefix:
            reversed = not reversed
        '''
        # get range and clamp
        minval, maxval = tuple(self.colorRange[key])
        if value <= minval: 
            fraction = 0.0
        elif value >= maxval:
            fraction = 1.0
        else:
            fraction = (value-minval)/(maxval-minval)

        # reverse if necessary
        if reversed:
            fraction = 1.0 - fraction

        # determine colors based on fraction
        color_scale = int(fraction*255*2)
        if color_scale > 255:
            r,g,b = (255*2-color_scale,255,0)
        else:
            r,g,b = (255,color_scale,0)

        # to hex
        hex_color = '#%02x%02x%02x' % (r,g,b)
        return hex_color


class APIExtractor(object):

    def __init__(self):
        pass

    @classmethod
    def cleanData(cls, data):
        if isinstance(data,dict):
            for k,v in data.iteritems():
                data[k] = cls.cleanData(v)
        elif isinstance(data, list):
            data = [cls.cleanData(_) for _ in data]
        elif APIExtractor.is_nan(data):
            data = None
        elif data == 'NaN':
            data = None
        elif isinstance(data, float):
            data = round(data,2)
        return data

    @classmethod
    def is_nan(cls, a):
        try:
            isnan = np.isnan(a)
            if isnan:
                return True
        except Exception as e:
            pass
        return False

    def fillDataRow(self, prefix, stats):
        row = {}
        for k,v in stats.iteritems():
            if APIExtractor.is_nan(v):
                row['%s_%s' % (prefix,k)] = 'NaN'
            else:
                row['%s_%s' % (prefix,k)] = v
        return row

    def refreshAPICacheValue(self, cache, key, row):
        try:
            item = cache[key]
            if item is None:
                item = {}
            for k,v in row.iteritems():
                item[k] = v
            cache[key] = item
            return True
        except Exception as e:
            print "Refresh %s Exception, key %s" % (cache, key)
            traceback.print_exc()
            return False

class TeamAPIExtractor(APIExtractor):

    @classmethod
    def generateCacheKey(cls, tid, gid):
        return '%s%s' % (tid, gid)

    @classmethod
    def getAPIResponseFromCache(cls, arg):
        tid, gid = arg
        key = cls.generateCacheKey(tid, gid)
        item = TEAM_REDIS_CACHE[key]
        if item is None:
            return None
        item = cls.cleanData(item)
        return item

    def fillBasicRow(self, vector):
        gamecode, abbr = vector.translateGameCode()
        row = {'gid': gamecode,
               'tid': abbr}
        return row

    def extractMetaVector(self, vector):
        stats = {}
        try:
            stats = {'days_rest': vector.days_rest,
                     'location': vector.location,
                     'home/road': vector.homeroad}
        except Exception as e:
            print "Meta Vector Exception: %s" % e
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('meta', stats))
        return row

    def extractSeasonVector(self, vector):
        stats = {}
        try:
            stats = vector.season_stats.loc['means'].to_dict()
        except Exception as e:
            print "Season Vector Exception: %s" % e
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('season', stats))
        return row

    def extractWindowVector(self, vector):
        stats = {}
        try:
            stats = vector.input.loc['means'].to_dict()
        except Exception as e:
            print "Window Vector Exception: %s" % e
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('windowed', stats))
        return row

    def extractVectors(self, vector, types, cache=True):
        whole_row = {}
        try:
            for t in types:
                if t == 'windowed':
                    row = self.extractWindowVector(vector)
                elif t == 'meta':
                    row = self.extractMetaVector(vector)
                elif t == 'season':
                    row = self.extractSeasonVector(vector)
                whole_row[t] = row
        except Exception as e:
            traceback.print_exc()
            raise e

        if cache:
            key = TeamAPIExtractor.generateCacheKey(vector.tid, vector.gid)
            self.refreshAPICacheValue(TEAM_REDIS_CACHE, key, whole_row)

        row_concat = {}
        for row in whole_row.values():
            row_concat.update(row)
        row_concat = TeamAPIExtractor.cleanData(row_concat)

        return row_concat


class PlayerAPIExtractor(APIExtractor):

    @classmethod
    def generateCacheKey(cls, pid, tid, gid):
        return '%s%s%s' % (pid, tid, gid)

    @classmethod
    def getAPIResponseFromCache(cls, arg):
        pid, tid, gid = arg
        key = cls.generateCacheKey(pid, tid, gid)
        item = PLAYER_REDIS_CACHE[key]
        if item is None:
            return item
        item = cls.cleanData(item)
        return item

    def fillBasicRow(self, vector):
        gamecode, abbr = vector.translateGameCode()
        row = {'gid': gamecode,
               'pid': vector.player_name,
               'tid': abbr}
        return row

    def extractMetaVector(self, vector):
        stats = {}
        try:
            stats = {'positions': '/'.join(vector.input['position']),
                     'days_rest': vector.input['days_rest'],
                     'location': vector.input['location'],
                     'home/road': vector.input['home/road']}
        except Exception as e:
            print "Meta Vector Exception: %s" % e
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('meta', stats))
        return row

    def extractWindowVector(self, vector):
        stats = {}
        try:
            stats = vector.input['own'].loc['means'].to_dict()
        except Exception as e:
            print "Window Vector Exception: %s" % e
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('windowed', stats))
        return row

    def extractExponentialVector(self, vector):
        stats = {}
        try:
            stats = vector.input['own'].loc['expmean'].to_dict()
        except Exception as e:
            print "Exp Vector Exception: %s" % e
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('exponential', stats))
        return row

    def extractOpponentPosVector(self, vector):
        stats = []
        try:
            stats = [v.loc['trend_mean'] for k,v in vector.input['against'].iteritems() if 'trend_mean' in v.index]
            stats = [_ for _ in stats if _ is not None]
        except Exception as e:
            print "Opp Pos Vector Exception: %s" % e
        if len(stats) > 0:
            stats_concat = pd.concat(tuple(stats), axis=1)
            stats_mean = dict(stats_concat.mean(axis=1))
        else:
            stats_mean = {}
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('opponent_pos', stats_mean))
        return row

    def extractTrendPosVector(self, vector):
        stats = []
        try:
            stats = [v.loc['mean'] for k,v in vector.input['trend'].iteritems() if 'mean' in v.index]
            stats = [_ for _ in stats if _ is not None]
        except Exception as e:
            print "Trend Exception: %s" % e
        if len(stats) > 0:
            stats_concat = pd.concat(tuple(stats), axis=1)
            stats_mean = dict(stats_concat.mean(axis=1))
        else:
            stats_mean = {}
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('trend_pos', stats_mean))
        return row

    def extractHomeRoadSplitVector(self, vector):
        stats = {}
        try:
            cur_game = vector.input['home/road'].lower()
            if cur_game == 'home':
                stats = vector.input['splits'].loc['home_split'].to_dict()
            else:
                stats = vector.input['splits'].loc['road_split'].to_dict()
        except Exception as e:
            print "Home Road Split Vector Exception: %s" % e
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('homeroadsplit', stats))
        return row

    def extractOppSplitVector(self, vector):
        stats = {}
        try:
            stats = vector.input['splits'].loc['against_team_split'].to_dict()
        except Exception as e:
            print "Opp Split Vector Exception: %s" % e
        row = self.fillBasicRow(vector)
        row.update(self.fillDataRow('oppsplit', stats))
        return row

    def extractVectors(self, vector, types, cache=True):
        whole_row = {}
        try:
            for t in types:
                if t == 'windowed':
                    row = self.extractWindowVector(vector)
                elif t == 'exponential':
                    row = self.extractExponentialVector(vector)
                elif t == 'meta':
                    row = self.extractMetaVector(vector)
                elif t == 'opponent_pos':
                    row = self.extractOpponentPosVector(vector)
                elif t == 'trend_pos':
                    row = self.extractTrendPosVector(vector)
                elif t == 'homeroadsplit':
                    row = self.extractHomeRoadSplitVector(vector)
                elif t == 'oppsplit':
                    row = self.extractOppSplitVector(vector)
                whole_row[t] = row
        except Exception as e:
            traceback.print_exc()
            raise e

        if cache:
            key = PlayerAPIExtractor.generateCacheKey(vector.pid, vector.tid, vector.gid)
            cached = self.refreshAPICacheValue(PLAYER_REDIS_CACHE,key,whole_row)
            print "caching: %s, result: %s" % (key,cached)

        row_concat = {}
        for row in whole_row.values():
            row_concat.update(row)
        row_concat = PlayerAPIExtractor.cleanData(row_concat)
        
        return row_concat

if __name__ == "__main__":
    from statsETL.bball.statsExtraction import getPlayerVector, getTeamVector

    arg = (203900, 1610612751, '0021500264')
    pid, tid, gid = arg

    vector = getPlayerVector(pid, tid, gid, recalculate=False)
    extractor = PlayerAPIExtractor()
    row = extractor.extractVectors(vector, PLAYER_TYPES_ALLOWED, cache=True)
    cached_response = PlayerAPIExtractor.getAPIResponseFromCache(arg)
    print cached_response
    sys.exit(1)

    # test team api response generation and caching
    team_vector = getTeamVector(tid, gid, recalculate=False)
    team_api_extractor = TeamAPIExtractor()
    row = team_api_extractor.extractVectors(team_vector, TEAM_TYPES_ALLOWED, cache=True)
    team_arg = (tid, gid)
    team_cache_response = TeamAPIExtractor.getAPIResponseFromCache(team_arg)
    print team_cache_response
