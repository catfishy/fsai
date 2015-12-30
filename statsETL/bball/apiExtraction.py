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
import re

import numpy as np
import pandas as pd
import pymongo
from pymongo.helpers import DuplicateKeyError

from statsETL.util.redislib import RedisTTLCache
from statsETL.db.mongolib import *


TEAM_REDIS_CACHE = RedisTTLCache('teamapi')
PLAYER_REDIS_CACHE = RedisTTLCache('playerapi')
PLAYER_TYPES_ALLOWED = ['windowed', 'opponent_pos', 'trend_pos', 'exponential', 'expdiff', 'homeroadsplit', 'oppsplit', 'meta']
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
                    'matchup': ['expdiff','trend_pos','opponent_pos','homeroadsplit','oppsplit']}
TEAM_STAT_TYPE = {'basic': ['windowed','meta'],
                  'matchup': ['windowed','season']}

STAT_TYPE_NAMES = {'expdiff': 'EXP MEAN DIFF',
                   'exponential': 'EXP',
                   'windowed': 'MEAN',
                   'meta': 'META',
                   'season': 'SEASON',
                   'opponent_pos': 'POS VS OPP',
                   'trend_pos': 'TREND IN POS',
                   'homeroadsplit': 'SPLIT HOME ROAD',
                   'oppsplit': 'SPLIT VS OPP',
                   'stat_key': 'STAT'}

'''
Specifies which stats to return, given a request type
'''
BASIC_STAT_TYPE_KEYS = [('gid','GAME'),
                        ('pid','PLAYER'),
                        ('positions','POS'),
                        ('days_rest','REST'),
                        ('MIN','MIN'),
                        ('USG_PCT','USG  %'),
                        ('TCHS','TCH'),
                        ('PTS','PTS'),
                        ('REB','REB'),
                        ('REB_PCT', 'REB  %'),
                        ('AST','AST'),
                        ('AST_RATIO','AST RATIO'),
                        ('TO','TO'),
                        ('STL','STL'),
                        ('BLK','BLK'),
                        ('FGA','FGA'),
                        ('TS_PCT','TS%'),
                        ('FG3A','3PA'),
                        ('FG3_PCT','3P%'),
                        ('FTA_RATE','FTA RATE'),
                        ('FT_PCT','FT%'),
                        ('PCT_AST_FGM','%  AST FGM'),
                        ('PCT_UAST_FGM','%  UAST FGM'),
                        ('PCT_FGM','%  FGM'),
                        ('PCT_PTS','%  PTS'),
                        ('PCT_REB','%  REB'),
                        ('PCT_AST','%  AST'),
                        ('PCT_BLK','%  BLK'),
                        ('PCT_FGA','%  FGA'),
                        ('PCT_FG3A','%  3PA'),
                        ('PCT_FTA','%  FTA'),
                        ('PCT_TOV','%  TO'),
                        ('PCT_STL','%  STL'),
                        ('PCT_PFD','%  PFD'),
                        ('PCT_BLKA','%  BLKA')]
MATCHUP_STAT_TYPE_KEYS = [('stat_key', 'STAT'),
                          ('PTS','PTS'),
                          ('AST','AST'),
                          ('REB','REB'),
                          ('BLK','BLK'),
                          ('STL','STL'),
                          ('PACE','PACE'),
                          ('PIE','PIE'),
                          ('FGA','FGA'),
                          ('TS_PCT','TS%'),
                          ('UFGA','UFGA'),
                          ('UFG_PCT','UFG%'),
                          ('CFGA','CFGA'),
                          ('CFG_PCT','CFG%'),
                          ('FG3A','3PA'),
                          ('FG3_PCT','3P%'),
                          ('AST_PCT','AST%'),
                          ('AST_TOV','AST/ TOV'),
                          ('OREB_PCT','OREB%'),
                          ('DREB_PCT','DREB%'),
                          ('TM_TOV_PCT','TOV%'),
                          ('PFD','PFD'),
                          ('FTA_RATE','FTA RATE'),
                          ('PTS_FB','FB PTS'),
                          ('PTS_PAINT','PAINT PTS'),
                          ('PTS_2ND_CHANCE','2ND CHANCE PTS'),
                          ('PTS_OFF_TOV','TOV PTS'),
                          ('Above_the_Break_3_Center(C)_24+_ft_attempted','3P (Center) Attmpt'),
                          ('Above_the_Break_3_Center(C)_24+_ft_percent','3P (Center) %'),
                          ('Above_the_Break_3_Right_Side_Center(RC)_24+_ft_attempted','3P (Right) Attmpt'),
                          ('Above_the_Break_3_Right_Side_Center(RC)_24+_ft_percent','3P (Right) %'),
                          ('Above_the_Break_3_Left_Side_Center(LC)_24+_ft_attempted','3P (Left) Attmpt'),
                          ('Above_the_Break_3_Left_Side_Center(LC)_24+_ft_percent','3P (Left) %'),
                          ('Left_Corner_3_Left_Side(L)_24+_ft_attempted','3P (Left Corner) Attmpt'),
                          ('Left_Corner_3_Left_Side(L)_24+_ft_percent','3P (Left Corner) %'),
                          ('Right_Corner_3_Right_Side(R)_24+_ft_attempted','3P (Right Corner) Attmpt'),
                          ('Right_Corner_3_Right_Side(R)_24+_ft_percent','3P (Right Corner) %'),
                          ('Mid_Range_Left_Side(L)_16_24_ft_attempted','MR (Left Long) Attmpt'),
                          ('Mid_Range_Left_Side(L)_16_24_ft_percent','MR (Left Long) %'),
                          ('Mid_Range_Left_Side_Center(LC)_16_24_ft_attempted','MR (Left Center Long) Attmpt'),
                          ('Mid_Range_Left_Side_Center(LC)_16_24_ft_percent','MR (Left Center Long) %'),
                          ('Mid_Range_Center(C)_16_24_ft_attempted','MR (Center Long) Attmpt'),
                          ('Mid_Range_Center(C)_16_24_ft_percent','MR (Center Long) %'),
                          ('Mid_Range_Right_Side_Center(RC)_16_24_ft_attempted','MR (Right Center Long) Attmpt'),
                          ('Mid_Range_Right_Side_Center(RC)_16_24_ft_percent','MR (Right Center Long) %'),
                          ('Mid_Range_Right_Side(R)_16_24_ft_attempted','MR (Right Long) Attmpt'),
                          ('Mid_Range_Right_Side(R)_16_24_ft_percent','MR (Right Long) %'),
                          ('Mid_Range_Left_Side(L)_8_16_ft_attempted','MR (Left Short) Attmpt'),
                          ('Mid_Range_Left_Side(L)_8_16_ft_percent','MR (Left Short) %'),
                          ('Mid_Range_Center(C)_8_16_ft_attempted','MR (Center Short) Attmpt'),
                          ('Mid_Range_Center(C)_8_16_ft_percent','MR (Center Short) %'),
                          ('Mid_Range_Right_Side(R)_8_16_ft_attempted','MR (Right Short) Attmpt'),
                          ('Mid_Range_Right_Side(R)_8_16_ft_percent','MR (Right Short) %'),
                          ('In_The_Paint_(Non_RA)_Left_Side(L)_8_16_ft_attempted','ITP (Left) Attmpt'),
                          ('In_The_Paint_(Non_RA)_Left_Side(L)_8_16_ft_percent','ITP (Left) %'),
                          ('In_The_Paint_(Non_RA)_Center(C)_8_16_ft_attempted','ITP (Center) Attmpt'),
                          ('In_The_Paint_(Non_RA)_Center(C)_8_16_ft_percent','ITP (Center) %'),
                          ('In_The_Paint_(Non_RA)_Right_Side(R)_8_16_ft_attempted','ITP (Right) Attmpt'),
                          ('In_The_Paint_(Non_RA)_Right_Side(R)_8_16_ft_percent','ITP (Right) %'),
                          ('In_The_Paint_(Non_RA)_Center(C)_Less_Than_8_ft_attempted','ITP (Center Low) Attmpt'),
                          ('In_The_Paint_(Non_RA)_Center(C)_Less_Than_8_ft_percent','ITP (Center Low) %'),
                          ('Restricted_Area_Center(C)_Less_Than_8_ft_attempted','RESTRICTED Attmpt'),
                          ('Restricted_Area_Center(C)_Less_Than_8_ft_percent','RESTRICTED %')]

def cleanKey(key):
    cleaned_key = key.replace('+','')
    cleaned_key = re.sub(r"\(.*?\)", "", cleaned_key)
    return cleaned_key


class colorExtractor(object):

    '''
    gradient from red to green
    '''

    def __init__(self, vector_type, date=None, gid=None):
        self.vector_type = vector_type
        self.colorRange = {}
        self.colorRangeDf = pd.DataFrame()
        if date is not None:
            self.loadColorRange(date)
        elif gid is not None:
            game_row = nba_conn.findByID(nba_games_collection, str(gid).zfill(10))
            if game_row is not None:
                self.loadColorRange(game_row['date'])
                print "Loaded color range for game %s" % gid 

    def updateColorRange(self, newvals, bin=None):
        if bin is None:
            bin = ['ALL']
        new_row = {'bin': bin}
        for k,v in newvals.iteritems():
            try:
                v = float(v)
            except Exception as e:
                continue # don't save non float values into color range
            new_row[k] = v
        self.colorRangeDf = pd.concat((self.colorRangeDf,pd.DataFrame([new_row])))
        self.colorRangeDf.reset_index(drop=True, inplace=True)
        if len(self.colorRangeDf.index) > 1500: # cap at 1500 rows
            self.colorRangeDf.drop(self.colorRangeDf.index[:1],inplace=True)

    def dateKey(self, date):
        return datetime(year=date.year, month=date.month, day=date.day)

    def saveColorRange(self, date):
        bins = list(set([b for l in np.unique(self.colorRangeDf.bin) for b in l]))
        if 'ALL' not in bins:
            bins.append('ALL')

        def indicator(bin, binlist):
            return bin in binlist

        for bin in bins:
            if bin == 'ALL':
                d = self.colorRangeDf.describe()
            else:
                in_bin = self.colorRangeDf.bin.apply(lambda row: indicator(bin,row))
                d = self.colorRangeDf[in_bin].describe()
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
            self.colorRange[bin] = {c: list(ranges.loc[:,c])  for c in ranges.columns}
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

    def extractColor(self, bin, key, value):
        # default transparent
        if bin not in self.colorRange:
            return 'transparent'
        if key not in self.colorRange[bin]:
            return 'transparent'
        if value is None:
            return 'transparent'

        # determine reversal
        prefix, suffix = APIExtractor.splitKey(key)
        reversed = True if suffix in REVERSED else False
        if reversed:
            print "REVERSED COLOR: %s" % key

        # in case there is a whole stat type where ranks are flipped
        '''
        if 'opponent' == prefix:
            reversed = not reversed
        '''
        # get range and clamp
        minval, maxval = tuple(self.colorRange[bin][key])
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
    def splitKey(cls, key):
        all_types = list(set(PLAYER_TYPES_ALLOWED) | set(TEAM_TYPES_ALLOWED))
        for t in all_types:
            if key.startswith(t):
                prefix = t
                suffix = key.replace('%s_' % t, '')
                return (prefix, suffix)
        return ('',key)

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
        row = {'gid': gamecode.replace(abbr, '<b>%s</b>' % abbr),
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
        row = {'gid': gamecode.replace(abbr, '<b>%s</b>' % abbr),
               'pid': vector.player_name,
               'tid': abbr}
        print row
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

    def extractExpDiffVector(self, vector, exp_row, window_row):
        get_val = lambda v: (np.nan if v == 'NaN' else v)
        exp_row = {k.replace('exponential_',''):get_val(v) for k,v in exp_row.iteritems()}
        window_row = {k.replace('windowed_',''):get_val(v) for k,v in window_row.iteritems()}
        df = pd.DataFrame([exp_row, window_row])
        to_drop = [_ for _ in df.columns if not all(df[_].apply(np.isreal))]
        df = df.drop(to_drop, axis=1)
        expdiff = df.loc[0].subtract(df.loc[1])
        row = self.fillDataRow('expdiff', dict(expdiff))
        # overwrite the nonsensical basic row values from diff operation
        row.update(self.fillBasicRow(vector)) 
        return row

    def extractVectors(self, vector, types, cache=True):
        whole_row = {}
        try:
            calc_expdiff = False
            for t in types:
                row = None
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
                elif t == 'expdiff':
                    calc_expdiff = True
                    continue
                whole_row[t] = row
            if calc_expdiff:
                exp_row = whole_row['exponential'] if 'exponential' in whole_row else self.extractExponentialVector(vector)
                wind_row = whole_row['windowed'] if 'windowed' in whole_row else self.extractWindowVector(vector)
                row = self.extractExpDiffVector(vector, exp_row, wind_row)
                whole_row['expdiff'] = row
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
