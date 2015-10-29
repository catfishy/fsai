from django.shortcuts import render
from rest_framework import permissions, viewsets, status, views
from rest_framework.response import Response
from datetime import datetime, timedelta
import json
from collections import defaultdict
import numpy as np
import copy

from statsETL.bball.statsExtraction import getTeamsForDay, getPlayersForDay, getPlayerVector, getTeamVector
from statsETL.util.crawler import daterange


'''
Return precalculated NBA player stat vectors in JSON format
'''

class DailyPlayerVectors(views.APIView):

    def get(self, request):
        return Response({}, status=status.HTTP_200_OK)

'''
Return precalculated NBA team stat vectors in JSON format
'''

class DailyTeamVectors(views.APIView):

    TYPES_ALLOWED = ['windowed', 'meta', 'season', 'opponent']
    KEY_NAMES = {'gid' : 'GAME',
                 'tid' : 'TEAM',
                 'NET_RATING' : 'Net Rating',
                 'PLUS_MINUS' : '+/-',
                 'PACE' : 'Pace',
                 'PIE' : 'PIE',
                 'PTS' : 'Points',
                 'PTS_PAINT' : 'Pts In Paint',
                 'PTS_OFF_TOV' : 'Pts Off TO',
                 'PTS_FB' : 'PTS_FB',
                 'PTS_2ND_CHANCE' : '2nd Chance Pts.',
                 'STL' : 'Steals',
                 'PASS' : 'Passes',
                 'AST' : 'Assists',
                 'SAST' : 'Secondary AST',
                 'AST_RATIO' : 'AST_RATIO',
                 'AST_PCT' : 'AST%',
                 'AST_TOV' : 'AST_TOV',
                 'TO' : 'TO',
                 'TM_TOV_PCT' : 'TOV%',
                 'FTA' : 'FTA',
                 'FT_PCT' : 'FT%',
                 'FTA_RATE' : 'FTA_RATE',
                 'FTAST' : 'FTAST',
                 'FGA' : 'FGA',
                 'FG_PCT' : 'FG%',
                 'DFGA' : 'DFGA',
                 'DFG_PCT' : 'DFG%',
                 'FG3A' : 'FG3A',
                 'FG3_PCT' : 'FG3%',
                 'TS_PCT' : 'TS%',
                 'EFG_PCT' : 'EFG%',
                 'UFGA' : 'Unassisted FGA',
                 'UFG_PCT' : 'Unassisted FG%',
                 'CFGA' : 'Contested FGA',
                 'CFG_PCT' : 'Contested FG%',
                 'REB' : 'REB',
                 'REB_PCT' : 'REB%',
                 'DREB' : 'DREB',
                 'DREB_PCT' : 'DREB%',
                 'OREB' : 'OREB',
                 'OREB_PCT' : 'OREB%',
                 'BLK' : 'BLK',
                 'BLKA' : 'BLKA',
                 'PCT_FGA_2PT' : 'PCT_FGA_2PT',
                 'PCT_FGA_3PT' : 'PCT_FGA_3PT',
                 'PCT_AST_2PM' : 'PCT_AST_2PM',
                 'PCT_AST_3PM' : 'PCT_AST_3PM',
                 'PCT_UAST_2PM' : 'PCT_UAST_2PM',
                 'PCT_UAST_3PM' : 'PCT_UAST_3PM',
                 'PCT_PTS_PAINT' : 'PCT_PTS_PAINT',
                 'PCT_PTS_2PT' : 'PCT_PTS_2PT',
                 'PCT_PTS_3PT' : 'PCT_PTS_3PT',
                 'PCT_PTS_FT' : 'PCT_PTS_FT',
                 'PCT_PTS_OFF_TOV' : 'PCT_PTS_OFF_TOV',
                 'PCT_PTS_FB' : 'PCT_PTS_FB',
                 'PCT_PTS_2PT_MR' : 'PCT_PTS_2PT_MR',
                 'OPP_FTA_RATE' : 'OPP_FTA_RATE',
                 'OPP_PTS_FB' : 'OPP_PTS_FB',
                 'OPP_EFG_PCT' : 'OPP eFG%',
                 'OPP_OREB_PCT' : 'OPP_OREB_PCT',
                 'OPP_PTS_PAINT' : 'OPP_PTS_PAINT',
                 'OPP_PTS_2ND_CHANCE' : 'OPP 2nd Chance Pts',
                 'OPP_TOV_PCT' : 'OPP TOV%',
                 'OPP_PTS_OFF_TOV' : 'OPP Pts Off TOV'}


    def extractMetaVector(self, vector):
        key_names = {'gid': 'GAME',
                     'tid': 'TEAM',
                     'home/road': 'Home/Road',
                     'location': 'Location',
                     'days_rest': 'Days Rest'}
        # get relevant stats to place in vector
        header = [{'key': k, 'name': n} for k,n in key_names.iteritems()]
        gamecode, abbr = vector.translateGameCode()
        row = {'gid': gamecode,
               'tid': abbr,
               'days_rest': vector.days_rest,
               'location': vector.input['location'],
               'home/road': vector.input['home/road']}
        return header, row

    def extractSeasonVector(self, vector):
        key_names = self.KEY_NAMES
        # get relevant stats to place in vector
        header = [{'key': k, 'name': n} for k,n in key_names.iteritems()]
        stats = vector.season_stats['means'].ix['mean'].to_dict()
        summary = vector.gamesummary
        gamecode, abbr = vector.translateGameCode()
        row = {'gid': gamecode,
               'tid': abbr}
        for k,v in stats.iteritems():
            if k not in key_names:
                continue
            if np.isnan(v):
                row[k] = 'NaN'
            else:
                row[k] = v
        return header, row

    def extractOppVector(self, vector):
        key_names = self.KEY_NAMES
        # get relevant stats to place in vector
        header = [{'key': k, 'name': n} for k,n in key_names.iteritems()]
        stats = vector.input['means_opp'].ix['mean'].to_dict()
        gamecode, abbr = vector.translateGameCode()
        row = {'gid': gamecode,
               'tid': abbr}
        for k,v in stats.iteritems():
            if k not in key_names:
                continue
            if np.isnan(v):
                row[k] = 'NaN'
            else:
                row[k] = v
        return header, row

    def extractOwnVector(self, vector):
        key_names = self.KEY_NAMES
        # get relevant stats to place in vector
        header = [{'key': k, 'name': n} for k,n in key_names.iteritems()]
        stats = vector.input['means'].ix['mean'].to_dict()
        gamecode, abbr = vector.translateGameCode()
        row = {'gid': gamecode,
               'tid': abbr}
        for k,v in stats.iteritems():
            if k not in key_names:
                continue
            if np.isnan(v):
                row[k] = 'NaN'
            else:
                row[k] = v
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
        data = {t: {'headers': [], 'rows': []} for t in types}
        for arg in args:
            tid, gid = arg
            vector = getTeamVector(tid, gid, recalculate=False)
            for t in types:
                if t == 'windowed':
                    header, row = self.extractOwnVector(vector)
                    data[t]['headers'] = header
                    data[t]['rows'].append(row)
                elif t == 'meta':
                    header, row = self.extractMetaVector(vector)
                    data[t]['headers'] = header
                    data[t]['rows'].append(row)
                elif t == 'season':
                    header, row = self.extractSeasonVector(vector)
                    data[t]['headers'] = header
                    data[t]['rows'].append(row)
                elif t == 'opponent':
                    header, row = self.extractOppVector(vector)
                    data[t]['headers'] = header
                    data[t]['rows'].append(row)

        # generate response
        data = dict(data)
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)
