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
TEAM_TYPES_ALLOWED = ['windowed', 'meta', 'season', 'opponent']

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

    def extractColor(self, key, value, reverse=False):
        if key not in self.colorRange:
            return '#000000'
        minval, maxval = tuple(self.colorRange[key])
        if value <= minval:
            fraction = 0.0
        elif value >= maxval:
            fraction = 1.0
        else:
            fraction = (value-minval)/(maxval-minval)

        if reverse:
            fraction = 1.0 - fraction
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

    def fillDataRow(self, prefix, key_names, stats):
        row = {}
        keys = dict(key_names).keys()
        for k,v in stats.iteritems():
            if k not in keys:
                continue
            if APIExtractor.is_nan(v):
                row['%s_%s' % (prefix,k)] = 'NaN'
            else:
                row['%s_%s' % (prefix,k)] = v
        return row

    def refreshAPICacheValue(self, cache, key, header, row):
        try:
            item = cache[key]
            if item is None:
                item = {}
            for k in header.keys():
                if k in item:
                    item[k]['headers'] = header[k]
                    item[k]['row'] = row[k]
                else:
                    item[k] = {'headers': header[k],
                               'row': row[k]}
            cache[key] = item
            return True
        except Exception as e:
            print "Refresh %s Exception, key %s" % (cache, key)
            traceback.print_exc()
            return False

class TeamAPIExtractor(APIExtractor):

    KEY_NAMES = [('NET_RATING','Net Rating'),
                 ('PLUS_MINUS','+/-'),
                 ('PACE','Pace'),
                 ('PIE','PIE'),
                 ('PTS','Points'),
                 ('PTS_PAINT','Pts In Paint'),
                 ('PTS_OFF_TOV','Pts Off TO'),
                 ('PTS_FB','PTS_FB'),
                 ('PTS_2ND_CHANCE','2nd Chance Pts.'),
                 ('STL','Steals'),
                 ('PASS','Passes'),
                 ('AST','Assists'),
                 ('SAST','Secondary AST'),
                 ('AST_RATIO','AST_RATIO'),
                 ('AST_PCT','AST%'),
                 ('AST_TOV','AST_TOV'),
                 ('TO','TO'),
                 ('TM_TOV_PCT','TOV%'),
                 ('FTA','FTA'),
                 ('FT_PCT','FT%'),
                 ('FTA_RATE','FTA_RATE'),
                 ('FTAST','FTAST'),
                 ('FGA','FGA'),
                 ('FG_PCT','FG%'),
                 ('DFGA','DFGA'),
                 ('DFG_PCT','DFG%'),
                 ('FG3A','FG3A'),
                 ('FG3_PCT','FG3%'),
                 ('TS_PCT','TS%'),
                 ('EFG_PCT','EFG%'),
                 ('UFGA','Unassisted FGA'),
                 ('UFG_PCT','Unassisted FG%'),
                 ('CFGA','Contested FGA'),
                 ('CFG_PCT','Contested FG%'),
                 ('REB','REB'),
                 ('REB_PCT','REB%'),
                 ('DREB','DREB'),
                 ('DREB_PCT','DREB%'),
                 ('OREB','OREB'),
                 ('OREB_PCT','OREB%'),
                 ('BLK','BLK'),
                 ('BLKA','BLKA'),
                 ('PCT_FGA_2PT','PCT_FGA_2PT'),
                 ('PCT_FGA_3PT','PCT_FGA_3PT'),
                 ('PCT_AST_2PM','PCT_AST_2PM'),
                 ('PCT_AST_3PM','PCT_AST_3PM'),
                 ('PCT_UAST_2PM','PCT_UAST_2PM'),
                 ('PCT_UAST_3PM','PCT_UAST_3PM'),
                 ('PCT_PTS_PAINT','PCT_PTS_PAINT'),
                 ('PCT_PTS_2PT','PCT_PTS_2PT'),
                 ('PCT_PTS_3PT','PCT_PTS_3PT'),
                 ('PCT_PTS_FT','PCT_PTS_FT'),
                 ('PCT_PTS_OFF_TOV','PCT_PTS_OFF_TOV'),
                 ('PCT_PTS_FB','PCT_PTS_FB'),
                 ('PCT_PTS_2PT_MR','PCT_PTS_2PT_MR'),
                 ('OPP_FTA_RATE','OPP_FTA_RATE'),
                 ('OPP_PTS_FB','OPP_PTS_FB'),
                 ('OPP_EFG_PCT','OPP eFG%'),
                 ('OPP_OREB_PCT','OPP_OREB_PCT'),
                 ('OPP_PTS_PAINT','OPP_PTS_PAINT'),
                 ('OPP_PTS_2ND_CHANCE','OPP 2nd Chance Pts'),
                 ('OPP_TOV_PCT','OPP TOV%'),
                 ('OPP_PTS_OFF_TOV','OPP Pts Off TOV')]

    @classmethod
    def generateCacheKey(cls, tid, gid):
        return '%s%s' % (tid, gid)

    @classmethod
    def getAPIResponseFromCache(cls, arg, types):
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

    def fillHeader(self, prefix, key_names):
        header = [{'key': 'gid', 'name': 'GAME'},
                  {'key': 'tid', 'name': 'TEAM'}]
        header += [{'key': '%s_%s' % (prefix,k), 'name': n} for k,n in key_names]
        return header

    def extractMetaVector(self, vector):
        key_names = [('home/road','Home/Road'),
                     ('location','Location'),
                     ('days_rest','Days Rest')]
        prefix = 'meta'
        header = self.fillHeader(prefix, key_names)
        row = self.fillBasicRow(vector)
        try:
            stats = {'days_rest': vector.days_rest,
                     'location': vector.input['location'],
                     'home/road': vector.input['home/road']}
        except Exception as e:
            print "Meta Vector Exception: %s" % e
            stats = {'days_rest': np.nan,
                     'location': np.nan,
                     'home/road': np.nan}
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractSeasonVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'season'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = vector.season_stats['means'].ix['mean'].to_dict()
        except Exception as e:
            print "Season Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractOppVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'opponent'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = vector.input['means_opp'].ix['mean'].to_dict()
        except Exception as e:
            print "Opp Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractWindowVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'windowed'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = vector.input['means'].ix['mean'].to_dict()
        except Exception as e:
            print "Window Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractVectors(self, vector, types, cache=True):
        whole_header = {}
        whole_row = {}
        try:
            for t in types:
                if t == 'windowed':
                    header, row = self.extractWindowVector(vector)
                elif t == 'meta':
                    header, row = self.extractMetaVector(vector)
                elif t == 'season':
                    header, row = self.extractSeasonVector(vector)
                elif t == 'opponent':
                    header, row = self.extractOppVector(vector)
                whole_header[t] = header
                whole_row[t] = row
        except Exception as e:
            traceback.print_exc()
            raise e

        if cache:
            key = TeamAPIExtractor.generateCacheKey(vector.tid, vector.gid)
            self.refreshAPICacheValue(TEAM_REDIS_CACHE, 
                                      key, 
                                      whole_header, 
                                      whole_row)

        row_concat = {}
        for row in whole_row.values():
            row_concat.update(row)
        row_concat = TeamAPIExtractor.cleanData(row_concat)
        return whole_header, row_concat


class PlayerAPIExtractor(APIExtractor):

    KEY_NAMES = [('MIN','MIN'),
                 ('USG_PCT','USG%'),
                 ('PTS','PTS'),
                 ('REB','REB'),
                 ('AST','AST'),
                 ('STL','STL'),
                 ('BLK','BLK'),
                 ('TO','TO'),
                 ('FGA','FGA'),
                 ('FG3A','FG3A'),
                 ('FTA_RATE','FTA Rate'),
                 ('FG_PCT','FG%'),
                 ('DFG_PCT','DFG%'),
                 ('TS_PCT','TS%'),
                 ('FT_PCT','FT%'),
                 ('REB_PCT','REB%'),
                 ('AST_PCT','AST%'),
                 ('AST_TOV','AST/TOV'),
                 ('PTS_PAINT','PTS PAINT'),
                 ('PTS_OFF_TOV','PTS OFF TOV'),
                 ('PTS_FB','PTS FB'),
                 ('PTS_2ND_CHANCE','PTS 2ND CHANCE'),
                 ('OPP_TOV_PCT','OPP TOV%'),
                 ('PCT_BLK','%BLK'),
                 ('PCT_FGA','%FGA'),
                 ('PCT_AST','%AST'),
                 ('PCT_STL','%STL'),
                 ('PCT_REB','%REB'),
                 ('OFF_RATING','Off Rate'),
                 ('DEF_RATING','Def Rate'),
                 ('NET_RATING','Net Rate'),
                 ('PIE','PIE'),
                 ('PLUS_MINUS','+/-'),
                 ('PACE','PACE'),
                 ('PFD','PFD'),
                 ('SPD','SPD'),
                 ('DIST','DIST'),
                 ('PASS','PASS'),
                 ('BLKA','BLKA'),
                 ('EFG_PCT','eFG%'),
                 ('FG3_PCT','FG3%'),
                 ('CFGA','CFGA'),
                 ('CFG_PCT','CFG%'),
                 ('UFGA','UFGA'),
                 ('UFG_PCT','UFG%'),
                 ('FTA','FTA'),
                 ('FTA_RATE','FTA Rate'),
                 ('SAST','SAST'),
                 ('FTAST','FTAST'),
                 ('AST_RATIO','AST RATIO'),
                 ('RBC','RBC'),
                 ('DREB','DREB'),
                 ('DRBC','DRBC'),
                 ('DREB_PCT','DREB%'),
                 ('OREB','OREB'),
                 ('ORBC','ORBC'),
                 ('OREB_PCT','OREB%'),
                 ('DFGA','DFGA'),
                 ('DFGM','DFGM'),
                 ('OPP_PTS_OFF_TOV','OPP PTS OFF TOV'),
                 ('OPP_PTS_2ND_CHANCE','OPP PTS 2ND CHANCE'),
                 ('OPP_EFG_PCT','OPP eFG%'),
                 ('OPP_PTS_PAINT','OPP PTS PAINT'),
                 ('OPP_PTS_FB','OPP PTS FB'),
                 ('OPP_OREB_PCT','OPP OREB%'),
                 ('OPP_FTA_RATE','OPP FTA RATE'),
                 ('PCT_AST_FGM','PCT_AST_FGM'),
                 ('PCT_DREB','PCT_DREB'),
                 ('PCT_AST_2PM','PCT_AST_2PM'),
                 ('PCT_UAST_3PM','PCT_UAST_3PM'),
                 ('PCT_PTS_FB','PCT_PTS_FB'),
                 ('PCT_PTS_2PT_MR','PCT_PTS_2PT_MR'),
                 ('PCT_PTS_FT','PCT_PTS_FT'),
                 ('PCT_PTS_2PT','PCT_PTS_2PT'),
                 ('PCT_AST_3PM','PCT_AST_3PM'),
                 ('PCT_PFD','PCT_PFD'),
                 ('PCT_PTS_PAINT','PCT_PTS_PAINT'),
                 ('PCT_UAST_2PM','PCT_UAST_2PM'),
                 ('PCT_PTS_OFF_TOV','PCT_PTS_OFF_TOV'),
                 ('PCT_PTS_3PT','PCT_PTS_3PT'),
                 ('PCT_BLKA','PCT_BLKA'),
                 ('PCT_FGA_3PT','PCT_FGA_3PT'),
                 ('PCT_FG3A','PCT_FG3A'),
                 ('PCT_FG3M','PCT_FG3M'),
                 ('PCT_PTS','PCT_PTS'),
                 ('PCT_FGM','PCT_FGM'),
                 ('PCT_FTM','PCT_FTM'),
                 ('PCT_FGA_2PT','PCT_FGA_2PT'),
                 ('PCT_FTA','PCT_FTA'),
                 ('PCT_TOV','PCT_TOV'),
                 ('PCT_OREB','PCT_OREB'),
                 ('PCT_UAST_FGM','PCT_UAST_FGM'),
                 ('PCT_PF','PCT_PF')]

    @classmethod
    def generateCacheKey(cls, pid, tid, gid):
        return '%s%s%s' % (pid, tid, gid)

    @classmethod
    def getAPIResponseFromCache(cls, arg, types):
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

    def fillHeader(self, prefix, key_names):
        header = [{'key': 'gid', 'name': 'GAME'},
                  {'key': 'tid', 'name': 'TEAM'},
                  {'key': 'pid', 'name': 'PLAYER'}]
        header += [{'key': '%s_%s' % (prefix,k), 'name': n} for k,n in key_names]
        return header

    def extractMetaVector(self, vector):
        key_names = [('home/road','Home/Road'),
                     ('positions','Positions'),
                     ('location','Location'),
                     ('days_rest','Days Rest')]
        prefix = 'meta'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = {'positions': vector.input['position'],
                     'days_rest': vector.input['days_rest'],
                     'location': vector.input['location'],
                     'home/road': vector.input['home/road']}
        except Exception as e:
            print "Meta Vector Exception: %s" % e
            stats = {'positions': np.nan,
                     'days_rest': np.nan,
                     'location': np.nan,
                     'home/road': np.nan}
        row = self.fillBasicRow(vector)
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractWindowVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'windowed'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = vector.input['means'].ix['mean'].to_dict()
        except Exception as e:
            print "Window Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractExponentialVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'exponential'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = vector.input['expmean'].to_dict()
        except Exception as e:
            print "Exp Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractOpponentPosVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'opponent_pos'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = [v['mean'].ix['mean'] for k,v in vector.input['against'].iteritems() if v]
        except Exception as e:
            print "Opp Pos Vector Exception: %s" % e
            stats = []
        row = self.fillBasicRow(vector)
        if len(stats) > 0:
            stats_concat = pd.concat(tuple(stats), axis=1)
            stats_mean = dict(stats_concat.mean(axis=1))
        else:
            stats_mean = {k: np.nan for k,n in key_names}
        row_data = self.fillDataRow(prefix, key_names, stats_mean)
        row.update(row_data)
        return header, row

    def extractTrendPosVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'trend_pos'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = [v['mean'].ix['mean'] for k,v in vector.input['trend'].iteritems() if v]
        except Exception as e:
            print "Trend Exception: %s" % e
            stats = []
        row = self.fillBasicRow(vector)
        if len(stats) > 0:
            stats_concat = pd.concat(tuple(stats), axis=1)
            stats_mean = dict(stats_concat.mean(axis=1))
        else:
            stats_mean = {k: np.nan for k,n in key_names}
        row_data = self.fillDataRow(prefix, key_names, stats_mean)
        row.update(row_data)
        return header, row

    def extractHomeRoadSplitVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'homeroadsplit'
        header = self.fillHeader(prefix, key_names)
        split_key = 'home_split' if vector.input['home/road'].lower() == 'home' else 'road_split'
        try:
            stats = vector.input['splits'][split_key].ix['mean'].to_dict()
        except Exception as e:
            print "Home Road Split Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractOppSplitVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'oppsplit'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = vector.input['splits']['against_team_split'].ix['mean'].to_dict()
        except Exception as e:
            print "Opp Split Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = self.fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractVectors(self, vector, types, cache=True):
        whole_header = {}
        whole_row = {}
        try:
            for t in types:
                if t == 'windowed':
                    header, row = self.extractWindowVector(vector)
                elif t == 'exponential':
                    header, row = self.extractExponentialVector(vector)
                elif t == 'meta':
                    header, row = self.extractMetaVector(vector)
                elif t == 'opponent_pos':
                    header, row = self.extractOpponentPosVector(vector)
                elif t == 'trend_pos':
                    header, row = self.extractTrendPosVector(vector)
                elif t == 'homeroadsplit':
                    header, row = self.extractHomeRoadSplitVector(vector)
                elif t == 'oppsplit':
                    header, row = self.extractOppSplitVector(vector)
                whole_header[t] = header
                whole_row[t] = row
        except Exception as e:
            traceback.print_exc()
            raise e

        if cache:
            key = PlayerAPIExtractor.generateCacheKey(vector.pid, vector.tid, vector.gid)
            cached = self.refreshAPICacheValue(PLAYER_REDIS_CACHE,key,whole_header,whole_row)
            print "caching: %s, result: %s" % (key,cached)
        row_concat = {}
        for row in whole_row.values():
            row_concat.update(row)
        row_concat = PlayerAPIExtractor.cleanData(row_concat)
        return whole_header, row_concat

if __name__ == "__main__":
    from statsETL.bball.statsExtraction import getPlayerVector, getTeamVector

    #arg = (1626170, 1610612752, u'0021500079')


    #arg = (201933,1610612746,u"0021500093")
    arg=(203099,1610612739,'0021500094')
    pid, tid, gid = arg
    vector = getPlayerVector(pid, tid, gid, recalculate=True)
    sys.exit(1)


    extractor = PlayerAPIExtractor()
    header, row = extractor.extractVectors(vector, PLAYER_TYPES_ALLOWED, cache=True)
    print header
    print row
    sys.exit(1)


    cached_response = PlayerAPIExtractor.getAPIResponseFromCache(arg, PLAYER_TYPES_ALLOWED)
    print cached_response
    sys.exit(1)


    # # test player api response generation and caching
    # player_vector = getPlayerVector(pid, tid, gid)
    # player_api_extractor = PlayerAPIExtractor()
    # print PlayerAPIExtractor.generateCacheKey(pid, tid, gid)
    # header, row = player_api_extractor.extractVectors(player_vector, PLAYER_TYPES_ALLOWED, cache=True)

    # # get from cache
    # player_arg = (pid, tid, gid)
    # player_cache_response = PlayerAPIExtractor.getAPIResponseFromCache(player_arg, PLAYER_TYPES_ALLOWED)
    # print player_cache_response

    # # test team api response generation and caching
    # team_vector = getTeamVector(tid, gid)
    # team_api_extractor = TeamAPIExtractor()
    # print TeamAPIExtractor.generateCacheKey(tid, gid)
    # header, row = team_api_extractor.extractVectors(team_vector, TEAM_TYPES_ALLOWED, cache=True)

    # team_arg = (tid, gid)
    team_arg = (1610612743,'0021500121')
    team_cache_response = TeamAPIExtractor.getAPIResponseFromCache(team_arg, TEAM_TYPES_ALLOWED)
    print team_cache_response
