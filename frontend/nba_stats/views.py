from django.shortcuts import render
from rest_framework import permissions, viewsets, status, views
from rest_framework.response import Response
from datetime import datetime, timedelta
import json
from collections import defaultdict
import numpy as np
import copy
import pandas as pd
import traceback

from statsETL.bball.statsExtraction import getTeamsForDay, getPlayersForDay, getPlayerVector, getTeamVector
from statsETL.util.crawler import daterange
from statsETL.util.redis import RedisTTLCache


def is_numeric(a):
   try:
       float(a)
   except Exception as e:
       return False
   return True

def fillDataRow(prefix, key_names, stats):
    row = {}
    keys = dict(key_names).keys()
    for k,v in stats.iteritems():
        if k not in keys:
            continue
        if is_numeric(v) and np.isnan(v):
            row['%s_%s' % (prefix,k)] = 'NaN'
        else:
            row['%s_%s' % (prefix,k)] = v
    return row

'''
Return precalculated NBA player stat vectors in JSON format
'''

class DailyPlayerVectors(views.APIView):

    REDIS_CACHE = RedisTTLCache('playerapi')
    TYPES_ALLOWED = ['windowed', 'opponent_pos', 'trend_pos', 'exponential', 'homeroadsplit', 'oppsplit', 'meta']
    KEY_NAMES = [('OFF_RATING','OFF_RATING'),
                 ('DEF_RATING','DEF_RATING'),
                 ('NET_RATING','NET_RATING'),
                 ('MIN','MIN'),
                 ('PACE','PACE'),
                 ('PIE','PIE'),
                 ('PLUS_MINUS','PLUS_MINUS'),
                 ('USG_PCT','USG_PCT'),
                 ('PFD','PFD'),
                 ('SPD','SPD'),
                 ('DIST','DIST'),
                 ('STL','STL'),
                 ('PASS','PASS'),
                 ('BLK','BLK'),
                 ('BLKA','BLKA'),
                 ('FGA','FGA'),
                 ('FG_PCT','FG_PCT'),
                 ('TS_PCT','TS_PCT'),
                 ('EFG_PCT','EFG_PCT'),
                 ('FG3A','FG3A'),
                 ('FG3_PCT','FG3_PCT'),
                 ('CFGA','CFGA'),
                 ('CFG_PCT','CFG_PCT'),
                 ('UFGA','UFGA'),
                 ('UFG_PCT','UFG_PCT'),
                 ('FTA','FTA'),
                 ('FT_PCT','FT_PCT'),
                 ('FTA_RATE','FTA_RATE'),
                 ('PTS','PTS'),
                 ('PTS_PAINT','PTS_PAINT'),
                 ('PTS_OFF_TOV','PTS_OFF_TOV'),
                 ('PTS_FB','PTS_FB'),
                 ('PTS_2ND_CHANCE','PTS_2ND_CHANCE'),
                 ('AST','AST'),
                 ('SAST','SAST'),
                 ('TO','TO'),
                 ('AST_TOV','AST_TOV'),
                 ('FTAST','FTAST'),
                 ('AST_PCT','AST_PCT'),
                 ('AST_RATIO','AST_RATIO'),
                 ('REB','REB'),
                 ('RBC','RBC'),
                 ('REB_PCT','REB_PCT'),
                 ('DREB','DREB'),
                 ('DRBC','DRBC'),
                 ('DREB_PCT','DREB_PCT'),
                 ('OREB','OREB'),
                 ('ORBC','ORBC'),
                 ('OREB_PCT','OREB_PCT'),
                 ('DFGA','DFGA'),
                 ('DFGM','DFGM'),
                 ('DFG_PCT','DFG_PCT'),
                 ('OPP_PTS_OFF_TOV','OPP_PTS_OFF_TOV'),
                 ('OPP_PTS_2ND_CHANCE','OPP_PTS_2ND_CHANCE'),
                 ('OPP_TOV_PCT','OPP_TOV_PCT'),
                 ('OPP_EFG_PCT','OPP_EFG_PCT'),
                 ('OPP_PTS_PAINT','OPP_PTS_PAINT'),
                 ('OPP_PTS_FB','OPP_PTS_FB'),
                 ('OPP_OREB_PCT','OPP_OREB_PCT'),
                 ('OPP_FTA_RATE','OPP_FTA_RATE'),
                 ('PCT_AST_FGM','PCT_AST_FGM'),
                 ('PCT_BLK','PCT_BLK'),
                 ('PCT_DREB','PCT_DREB'),
                 ('PCT_AST_2PM','PCT_AST_2PM'),
                 ('PCT_UAST_3PM','PCT_UAST_3PM'),
                 ('PCT_PTS_FB','PCT_PTS_FB'),
                 ('PCT_PTS_2PT_MR','PCT_PTS_2PT_MR'),
                 ('PCT_PTS_FT','PCT_PTS_FT'),
                 ('PCT_PTS_2PT','PCT_PTS_2PT'),
                 ('PCT_FGA','PCT_FGA'),
                 ('PCT_AST_3PM','PCT_AST_3PM'),
                 ('PCT_PFD','PCT_PFD'),
                 ('PCT_PTS_PAINT','PCT_PTS_PAINT'),
                 ('PCT_UAST_2PM','PCT_UAST_2PM'),
                 ('PCT_PTS_OFF_TOV','PCT_PTS_OFF_TOV'),
                 ('PCT_PTS_3PT','PCT_PTS_3PT'),
                 ('PCT_BLKA','PCT_BLKA'),
                 ('PCT_FGA_3PT','PCT_FGA_3PT'),
                 ('PCT_AST','PCT_AST'),
                 ('PCT_FG3A','PCT_FG3A'),
                 ('PCT_FG3M','PCT_FG3M'),
                 ('PCT_PTS','PCT_PTS'),
                 ('PCT_FGM','PCT_FGM'),
                 ('PCT_FTM','PCT_FTM'),
                 ('PCT_FGA_2PT','PCT_FGA_2PT'),
                 ('PCT_FTA','PCT_FTA'),
                 ('PCT_TOV','PCT_TOV'),
                 ('PCT_STL','PCT_STL'),
                 ('PCT_REB','PCT_REB'),
                 ('PCT_OREB','PCT_OREB'),
                 ('PCT_UAST_FGM','PCT_UAST_FGM'),
                 ('PCT_PF','PCT_PF')]

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
        row_data = fillDataRow(prefix, key_names, stats)
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
        row_data = fillDataRow(prefix, key_names, stats)
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
        row_data = fillDataRow(prefix, key_names, stats)
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
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        if len(stats) > 0:
            stats_concat = pd.concat(tuple(stats), axis=1)
            stats_mean = dict(stats_concat.mean(axis=1))
            row_data = fillDataRow(prefix, key_names, stats_mean)
        else:
            row_data = {'%s_%s' % (prefix,k) : 0.0 for k,n in key_names}
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
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        if len(stats) > 0:
            stats_concat = pd.concat(tuple(stats), axis=1)
            stats_mean = dict(stats_concat.mean(axis=1))
            row_data = fillDataRow(prefix, key_names, stats_mean)
        else:
            row_data = {'%s_%s' % (prefix,k) : 0.0 for k,n in key_names}
        return header, row

    def extractHomeRoadSplitVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'homeroadsplit'
        header = self.fillHeader(prefix, key_names)
        split_key = 'home_split' if vector.input['home/road'].lower() == 'home' else 'road_split'
        try:
            stats = vector.input['splits'][split_key].ix['mean'].to_dict()
        except Exception as e:
            print "Split Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def extractOppSplitVector(self, vector):
        key_names = self.KEY_NAMES
        prefix = 'oppsplit'
        header = self.fillHeader(prefix, key_names)
        try:
            stats = vector.input['splits']['against_team_split'].ix['mean'].to_dict()
        except Exception as e:
            print "Split Vector Exception: %s" % e
            stats = {k: np.nan for k,n in key_names}
        row = self.fillBasicRow(vector)
        row_data = fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def get(self, request):
        params = self.request.GET
        if 'from' not in params or 'to' not in params:
            return Response({'status': 'Bad request', 'message': 'Missing date parameters'}, status=status.HTTP_400_BAD_REQUEST)

        # generate stats query
        try:
            from_date = datetime.strptime(params['from'],'%Y-%m-%d')
            to_date = datetime.strptime(params['to'], '%Y-%m-%d') + timedelta(days=1)
            types = [_ for _ in params['types'].split(',') if _]
            if len(set(types) - set(self.TYPES_ALLOWED)) > 0:
                raise Exception("Invalid Type")
        except Exception as e:
            return Response({'status': 'Bad request','message': 'Invalid format: %s' % e}, status=status.HTTP_400_BAD_REQUEST)
        args = []
        for date in daterange(from_date, to_date):
            new_args = getPlayersForDay(date)
            args += new_args

        # query for stats and load response data
        data = {'headers' : {}, 'rows': []}
        row_dict = defaultdict(dict)
        for arg in args:
            pid, tid, gid = arg
            vector = getPlayerVector(pid, tid, gid, recalculate=False)
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
                    data['headers'][t] = header
                    row_dict[(pid, tid, gid)].update(row)
            except Exception as e:
                traceback.print_exc()
                raise e
        data['rows'] = row_dict.values()

        # generate response
        data = dict(data)
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)

'''
Return precalculated NBA team stat vectors in JSON format
'''

class DailyTeamVectors(views.APIView):

    REDIS_CACHE = RedisTTLCache('teamapi')
    TYPES_ALLOWED = ['windowed', 'meta', 'season', 'opponent']
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

    def fillDataRow(self, prefix, key_names, stats):
        row = {}
        keys = dict(key_names).keys()
        for k,v in stats.iteritems():
            if k not in keys:
                continue
            if is_numeric(v) and np.isnan(v):
                row['%s_%s' % (prefix,k)] = 'NaN'
            else:
                row['%s_%s' % (prefix,k)] = v
        return row

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
        row_data = fillDataRow(prefix, key_names, stats)
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
        row_data = fillDataRow(prefix, key_names, stats)
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
        row_data = fillDataRow(prefix, key_names, stats)
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
        row_data = fillDataRow(prefix, key_names, stats)
        row.update(row_data)
        return header, row

    def get(self, request):
        params = self.request.GET
        if 'from' not in params or 'to' not in params:
            return Response({'status': 'Bad request', 'message': 'Missing date parameters'}, status=status.HTTP_400_BAD_REQUEST)

        # generate stats query
        try:
            from_date = datetime.strptime(params['from'],'%Y-%m-%d')
            to_date = datetime.strptime(params['to'], '%Y-%m-%d') + timedelta(days=1)
            types = [_ for _ in params['types'].split(',') if _]
            if len(set(types) - set(self.TYPES_ALLOWED)) > 0:
                raise Exception("Invalid Type")
        except Exception as e:
            return Response({'status': 'Bad request','message': 'Invalid format: %s' % e}, status=status.HTTP_400_BAD_REQUEST)
        args = []
        for date in daterange(from_date, to_date):
            new_args = getTeamsForDay(date)
            args += new_args

        # query for stats and load response data
        data = {'headers' : {}, 'rows': []}
        row_dict = defaultdict(dict)
        for arg in args:
            tid, gid = arg
            vector = getTeamVector(tid, gid, recalculate=False)
            for t in types:
                if t == 'windowed':
                    header, row = self.extractWindowVector(vector)
                elif t == 'meta':
                    header, row = self.extractMetaVector(vector)
                elif t == 'season':
                    header, row = self.extractSeasonVector(vector)
                elif t == 'opponent':
                    header, row = self.extractOppVector(vector)
                data['headers'][t] = header
                row_dict[(tid, gid)].update(row)
        data['rows'] = row_dict.values()

        # generate response
        data = dict(data)
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)
