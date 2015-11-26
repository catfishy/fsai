from django.shortcuts import render
from rest_framework import permissions, viewsets, status, views
from rest_framework.response import Response
from datetime import datetime, timedelta
import json
from collections import defaultdict

from statsETL.bball.apiExtraction import colorExtractor, TeamAPIExtractor, PlayerAPIExtractor, PLAYER_TYPES_ALLOWED, TEAM_TYPES_ALLOWED
from statsETL.bball.statsExtraction import getTeamsForDay, getPlayersForDay, getPlayerVector, getTeamVector
from statsETL.util.crawler import daterange

'''
Return precalculated NBA player stat vectors in JSON format
'''

class DailyPlayerVectors(views.APIView):

    REVERSED=[]


    def get(self, request):
        params = self.request.GET
        if 'from' not in params or 'to' not in params:
            return Response({'status': 'Bad request', 'message': 'Missing date parameters'}, status=status.HTTP_400_BAD_REQUEST)

        # generate stats query
        try:
            from_date = datetime.strptime(params['from'],'%Y-%m-%d')
            to_date = datetime.strptime(params['to'], '%Y-%m-%d') + timedelta(days=1)
            types = [_ for _ in params['types'].split(',') if _]
            if len(set(types) - set(PLAYER_TYPES_ALLOWED)) > 0:
                raise Exception("Invalid Type specified")
            if len(set(types) & set(PLAYER_TYPES_ALLOWED)) == 0:
                raise Exception("No valid types specified")
        except Exception as e:
            return Response({'status': 'Bad request','message': 'Invalid format: %s' % e}, status=status.HTTP_400_BAD_REQUEST)
        args = []
        for date in daterange(from_date, to_date):
            new_args = getPlayersForDay(date)
            args += new_args

        color = colorExtractor('player', date=to_date)

        # query for stats and load response data
        data = {'headers' : {}, 'rows': []}
        row_dict = defaultdict(dict)
        for arg in args:
            types_left = set(types)

            # look in cache
            cached_response = PlayerAPIExtractor.getAPIResponseFromCache(arg, types)
            if cached_response is not None:
                for t,v in cached_response.iteritems():
                    if t in types_left:
                        data['headers'][t] = v['headers']
                        row_dict[arg].update(v['row'])
                        types_left.discard(t)
            else:
                print "NO CACHED RESPONSE FOR %s" % (arg,)
            
            # extract if necessary
            if len(types_left) > 0:
                pid, tid, gid = arg
                vector = getPlayerVector(pid, tid, gid, recalculate=False)
                extractor = PlayerAPIExtractor()
                header, row = extractor.extractVectors(vector, types, cache=True)
                data['headers'] = header
                row_dict[arg] = row

            # remove row if empty windowed stats
            if 'windowed' in types:
                windowed_keys = [_['key'] for _ in data['headers']['windowed'] if 'windowed' in _['key']]
                windowed_values = [row_dict[arg][_] is None for _ in windowed_keys]
                if all(windowed_values):
                    row_dict.pop(arg)

            # fill colors
            if arg in row_dict:
                row = row_dict[arg]
                for k,v in row.iteritems():
                    row[k] = {'v':v, 'c': color.extractColor(k, v, reverse=(k in self.REVERSED))}
                row_dict[arg] = row

        # generate response
        data['rows'] = row_dict.values()
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)

'''
Return precalculated NBA team stat vectors in JSON format
'''

class DailyTeamVectors(views.APIView):

    TYPES_ALLOWED = ['windowed', 'meta', 'season', 'opponent']

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

        color = colorExtractor('player', date=to_date)

        # query for stats and load response data
        data = {'headers' : {}, 'rows': []}
        row_dict = defaultdict(dict)
        for arg in args:
            types_left = set(types)

            # look in cache
            cached_response = TeamAPIExtractor.getAPIResponseFromCache(arg, types)
            if cached_response is not None:
                for t,v in cached_response.iteritems():
                    if t in types_left:
                        data['headers'][t] = v['headers']
                        row_dict[arg].update(v['row'])
                        types_left.discard(t)
            if len(types_left) == 0:
                continue
            
            # extract
            tid, gid = arg
            vector = getTeamVector(tid, gid, recalculate=False)
            extractor = TeamAPIExtractor()
            header, row = extractor.extractVectors(vector, types, cache=True)
            data['headers'] = headers
            row_dict[arg] = row

            # fill colors
            if arg in row_dict:
                row = row_dict[arg]
                for k,v in row.iteritems():
                    row[k] = {'v':v, 'c': color.extractColor(k, v, reverse=(k in self.REVERSED))}
                row_dict[arg] = row

        # generate response
        data['rows'] = row_dict.values()
        data = dict(data)
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)
